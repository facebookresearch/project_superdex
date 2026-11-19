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
#include "rendering/deformable_surface.h"
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
class LineSegmentMesh;
class MaterialInstance;
class Mesh;
class ResourceManager;
class WireframeMesh;
} // namespace mochi_renderer

namespace superdex::studio {

class AssetManager;

// Role suffix hashed with a mesh's color seed to color the SDF surface reconstructed from it (see
// MochiModelAsset::GetSdfSurfaceColor), so SDF surfaces read alike everywhere they are shown.
inline constexpr std::string_view kSdfSurfaceColorRole = " output SDF";

// The material of a polyline (rod) centerline: unlit, and drawn over everything rather than depth
// tested, since the centerline runs inside the rod's tube and a straight rod lying flat sits
// exactly in the viewport's ground plane.
std::shared_ptr<mochi_renderer::MaterialInstance> CreatePolylineMaterial(
    mochi_renderer::ResourceManager const& resourceManager,
    filament::math::float3 color,
    float opacity = 1.0f);

//--------------------------------------------------------------------------------------------------
// MOCHI MODEL ASSET
//--------------------------------------------------------------------------------------------------

class MochiModelAsset : public Asset {
 public:
  ~MochiModelAsset() override;
  // An instance of the model's shared render mesh: its centerline for a polyline (rod) model, else
  // its flat-lit surface with a wireframe overlay.
  std::unique_ptr<mochi_renderer::SceneObject> GetRenderModelInstance() const;
  // True if the model's mesh is a polyline (rod) mesh.
  bool IsPolyline() const;
  // The element frame axes of a polyline model as Mochi creates its shape: the authored axes, or
  // else the discrete Bishop frame Mochi generates (3 values per element). Valid until the model is
  // reloaded; empty if the model is not a polyline or its shape cannot be created.
  mochi::Span<mochi::real const> GetElementFrameAxes();
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
  // Node count of one of the shape's meshes, regardless of element type, or 0 if the shape has no
  // such mesh. DeformableSurface::Surface is the tetrahedral boundary or the triangle surface
  // itself; DeformableSurface::Centerline is a polyline itself.
  int GetSurfaceVertexCount(DeformableSurface surface);
  // The meshes a soft actor of this model is drawn with (see ChooseSoftSurfaces).
  DeformableSurfaces GetSoftSurfaces(bool useContactSkin) const;
  // The meshes a rod actor of this model is drawn with (see ChooseRodSurfaces).
  DeformableSurfaces GetRodSurfaces(bool useContactSkin) const;
  // A wireframe of `surfaces.collision`, plus a solid mesh of `surfaces.rendered` when
  // @p createSolid is true, both deformable. A DeformableSurface::Centerline surface gets no mesh
  // here; a rod's centerline is a CreateCenterlineMesh line instead.
  bool CreateDeformableMeshes(
      DeformableSurfaces const& surfaces,
      mochi::Real3 const& bakeScale,
      mochi::TransformRT const& shapeTransform,
      bool createSolid,
      std::unique_ptr<mochi_renderer::Mesh>& outSolid,
      std::unique_ptr<mochi_renderer::WireframeMesh>& outWireframe);
  // Restores meshes made by CreateDeformableMeshes with the same `surfaces` to their rest pose.
  void UpdateDeformableMeshes(
      DeformableSurfaces const& surfaces,
      mochi_renderer::Mesh* solid,
      mochi_renderer::WireframeMesh* wireframe,
      mochi::Real3 const& bakeScale,
      mochi::TransformRT const& shapeTransform);
  // A deformable line mesh of a polyline model's centerline, baked like CreateDeformableMeshes;
  // null if the model is not a polyline.
  std::unique_ptr<mochi_renderer::LineSegmentMesh> CreateCenterlineMesh(
      mochi::Real3 const& bakeScale,
      mochi::TransformRT const& shapeTransform);
  // Restores a CreateCenterlineMesh line to its rest pose.
  void UpdateCenterlineMesh(
      mochi_renderer::LineSegmentMesh& centerline,
      mochi::Real3 const& bakeScale,
      mochi::TransformRT const& shapeTransform);
  // Sets @p renderModel, a render model linked to this model's visual mesh (see
  // CheckRenderModelLink), to the visual mesh's rest pose, baked like UpdateDeformableMeshes.
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

  // Load one of the shape's meshes (native, unbaked). DeformableSurface::Surface is the tetrahedral
  // boundary for a tet mesh, or the triangle surface itself for a surface mesh;
  // DeformableSurface::Centerline is the shape's own (polyline) mesh. Vertex ordering matches the
  // physics engine's node data for that mesh (e.g. SurfaceNodePositions), so the same buffer can be
  // updated in place during simulation. Empty if the shape has no such mesh.
  mochi::MeshDataView GetNativeSurfaceMesh(DeformableSurface surface, mochi::Error& error);
  // Load the model's tetrahedral boundary surface (native, unbaked); empty view + one warning if
  // the model is not a tetrahedral mesh.
  mochi::MeshDataView GetNativeSoftSurface(mochi::Error& error);
  // Bake a native triangle mesh for `bakeScale` + `shapeTransform` into flat renderer-space buffers
  // (area-weighted normals). Works for both tetrahedral boundaries and triangle surfaces (soft
  // callers are already gated to tet meshes upstream). Returns false if the shape has no such mesh.
  bool BakeDeformableSurface(
      DeformableSurface surface,
      mochi::Real3 const& bakeScale,
      mochi::TransformRT const& shapeTransform,
      std::vector<float>& positions,
      std::vector<float>& normals,
      std::vector<int>& indices);
  // Bakes @p mesh's nodes for `bakeScale` + `shapeTransform` into flat renderer-space positions.
  void BakeNodePositions(
      mochi::MeshDataView const& mesh,
      mochi::Real3 const& bakeScale,
      mochi::TransformRT const& shapeTransform,
      std::vector<float>& positions) const;
  // The native centerline of a polyline model, with its nodes baked into @p positions like
  // BakeNodePositions; empty if the model is not a polyline.
  mochi::MeshDataView BakeCenterline(
      mochi::Real3 const& bakeScale,
      mochi::TransformRT const& shapeTransform,
      std::vector<float>& positions);
  // Read the (surface-aligned) skinning from the native surface mesh into per-node bone indices and
  // weights (numNodes * weightsPerNode each, in the same node order as BakeDeformableSurface).
  // Returns false if the shape carries no surface skinning. `outMaxBoneIndex` is the largest bone
  // index referenced (-1 if none).
  bool BakeDeformableSurfaceSkinning(
      std::vector<int>& boneIndices,
      std::vector<float>& boneWeights,
      int& weightsPerNode,
      int& outMaxBoneIndex);
  // Caches @p surface as the built SDF surface and points the shared SDF mesh (and so every live
  // instance) at it.
  void InstallSdfSurface(std::optional<SdfSurface> surface);

  // Render geometry in renderer space: triangles with normals, or (for a polyline model) line
  // segments with no normals.
  struct RenderGeometry {
    bool isPolyline = false;
    std::vector<float> positions;
    std::vector<float> normals;
    std::vector<int> indices;
  };
  // Returns false if @p model has no usable render geometry.
  static bool BuildRenderGeometry(
      mochi::ModelData const& model,
      mochi::CoordinateSpaceConverter const* converter,
      RenderGeometry& out);
  // Points the shared render mesh of @p geometry's kind (and so every live instance) at it,
  // creating the mesh if needed.
  bool InstallRenderGeometry(
      RenderGeometry const& geometry,
      mochi_renderer::ResourceManager& resourceManager);

 private:
  std::unique_ptr<mochi_renderer::WireframeMesh> _renderModel;
  std::unique_ptr<mochi_renderer::LineSegmentMesh> _polylineRenderModel;
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
