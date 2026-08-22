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

#include <mochi_renderer/material.h>
#include <mochi_renderer/resource.h>
#include <mochi_renderer/scene_object.h>

#include <filament/Box.h>
#include <filament/Engine.h>
#include <mochi_core/utils/span.h>

#include <utils/Entity.h>

#include <vector>

//------------------------------------------------------------------------------------------------
// UTILS
//------------------------------------------------------------------------------------------------

namespace mochi {
struct ModelData;
class CoordinateSpaceConverter;
} // namespace mochi

namespace mochi_renderer {

// A freshly created static VertexBuffer/IndexBuffer pair plus the bounds of the
// geometry they hold. Ownership of the buffers is transferred to the caller, who
// must destroy them via the same @ref filament::Engine that created them.
struct MeshBuffers {
  filament::VertexBuffer* vertexBuffer = nullptr;
  filament::IndexBuffer* indexBuffer = nullptr;
  filament::Box bounds;
};

// Creates a static (GPU-immutable) VertexBuffer (POSITION FLOAT3 @ slot 0,
// normalized TANGENTS SHORT4 @ slot 1) and a UINT IndexBuffer from raw mesh data,
// packing the supplied per-vertex normals into the tangent quaternion attribute via
// filament::geometry::SurfaceOrientation. The returned bounds are the axis-aligned
// box of `positions`.
//
// @param engine     Filament engine that creates (and later destroys) the buffers.
// @param positions  Flat array of vertex positions (x, y, z, ...).
// @param normals    Flat array of vertex normals matching `positions`.
// @param indices    Triangle indices into the vertex arrays.
MeshBuffers CreateStaticMeshBuffers(
    filament::Engine& engine,
    mochi::Span<float const> positions,
    mochi::Span<float const> normals,
    mochi::Span<int const> indices);

// Like @ref CreateStaticMeshBuffers, but additionally declares the dummy COLOR/UV0/UV1
// vertex attributes that gltfio's ubershader material requires (all three sharing one
// constant buffer). Use this for geometry that replaces a gltfio-loaded asset's buffers
// (e.g. @ref RenderModel), so the material's required-attribute set is satisfied.
MeshBuffers CreateModelMeshBuffers(
    filament::Engine& engine,
    mochi::Span<float const> positions,
    mochi::Span<float const> normals,
    mochi::Span<int const> indices);

// Extracts renderable triangle geometry from a @ref mochi::ModelData, converting it
// from Mochi space into the renderer's space (@ref RenderSpace).
//
// For a tetrahedral mesh the boundary surface is extracted; a triangle surface mesh is
// used directly; otherwise a procedural box/plane/sphere is generated from the model's
// analytic shape. Per-vertex normals are angle-weighted (mesh) or supplied by the
// generator (procedural). Positions/normals are transformed by `converter`; when
// `converter` is null a default Mochi-to-@ref RenderSpace
// @ref mochi::CoordinateSpaceConverter is used. Triangle winding is left unchanged.
//
// @return false (leaving the outputs unspecified) if the model has no usable geometry.
bool BuildMochiModelGeometry(
    mochi::ModelData const& modelData,
    mochi::CoordinateSpaceConverter const* converter,
    std::vector<float>& positions,
    std::vector<float>& normals,
    std::vector<int>& indices);

//------------------------------------------------------------------------------------------------
// MESH
//------------------------------------------------------------------------------------------------

class MeshInstance;

class Mesh : public SceneObject, public IInstanceable {
 public:
  static std::unique_ptr<Mesh> CreateMesh(
      filament::Engine* engine,
      mochi::Span<float const> positions,
      mochi::Span<float const> normals,
      mochi::Span<int const> indices,
      std::shared_ptr<MaterialInstance> material,
      bool isDynamic = false,
      bool isClosed = true);
  ~Mesh() override;

  bool IsDynamic() const;
  bool Update(mochi::Span<float const> positions, mochi::Span<float const> normals);
  void SetMaterial(std::shared_ptr<MaterialInstance> material) override;
  utils::Entity GetRootEntity() const override;
  mochi::Span<utils::Entity const> GetEntities() const override;

  std::unique_ptr<SceneObject> GetInstance() override;
  int GetInstanceCount() const override;
  IInstanceable* GetInstanceable() override;

 private:
  Mesh(
      filament::Engine* engine,
      mochi::Span<float const> positions,
      mochi::Span<float const> normals,
      mochi::Span<int const> indices,
      std::shared_ptr<MaterialInstance> material,
      bool isDynamic = false,
      bool isClosed = true);

 private:
  friend class MeshInstance;
  bool _isDynamic = false;
  bool _isClosed = true;
  size_t _vertexCount = 0;
  size_t _indexCount = 0;
  utils::Entity _entity;
  filament::VertexBuffer* _vertexBuffer = nullptr;
  filament::IndexBuffer* _indexBuffer = nullptr;
  filament::Box _boundingBox;
  std::shared_ptr<MaterialInstance> _material = nullptr;
  int _instanceCount = 0;

  std::vector<filament::math::float3> _dynamicPositions;
  std::vector<filament::math::short4> _dynamicTangents;
};

class MeshInstance : public SceneObject {
 public:
  ~MeshInstance() override;
  utils::Entity GetRootEntity() const override;
  mochi::Span<utils::Entity const> GetEntities() const override;
  IInstanceable* GetInstanceable() override;

 private:
  friend class Mesh;
  Mesh* _primaryMesh;
  MeshInstance(filament::Engine* engine, utils::Entity entity, Mesh* primaryMesh);
  utils::Entity _entity;
};

//------------------------------------------------------------------------------------------------
// LINE MESH
//------------------------------------------------------------------------------------------------

class LineSegmentMeshInstance;

class LineSegmentMesh : public SceneObject, public IInstanceable {
 public:
  static std::unique_ptr<LineSegmentMesh> CreateLineSegmentMesh(
      filament::Engine* engine,
      mochi::Span<float const> positions,
      mochi::Span<int const> indices,
      std::shared_ptr<MaterialInstance> material,
      bool isDynamic = false);
  ~LineSegmentMesh() override;

  bool IsDynamic() const;
  bool Update(mochi::Span<float const> positions);
  void SetMaterial(std::shared_ptr<MaterialInstance> material) override;
  utils::Entity GetRootEntity() const override;
  mochi::Span<utils::Entity const> GetEntities() const override;

  std::unique_ptr<SceneObject> GetInstance() override;
  int GetInstanceCount() const override;
  IInstanceable* GetInstanceable() override;

 private:
  LineSegmentMesh(
      filament::Engine* engine,
      mochi::Span<float const> positions,
      mochi::Span<int const> indices,
      std::shared_ptr<MaterialInstance> material,
      bool isDynamic = false);

 private:
  friend class LineSegmentMeshInstance;
  bool _isDynamic = false;
  size_t _vertexCount = 0;
  size_t _indexCount = 0;
  utils::Entity _entity;
  filament::VertexBuffer* _vertexBuffer = nullptr;
  filament::IndexBuffer* _indexBuffer = nullptr;
  filament::Box _boundingBox;
  std::shared_ptr<MaterialInstance> _material = nullptr;
  int _instanceCount = 0;

  std::vector<filament::math::float3> _dynamicPositions;
};

class LineSegmentMeshInstance : public SceneObject {
 public:
  ~LineSegmentMeshInstance() override;
  utils::Entity GetRootEntity() const override;
  mochi::Span<utils::Entity const> GetEntities() const override;
  IInstanceable* GetInstanceable() override;

 private:
  friend class LineSegmentMesh;
  LineSegmentMesh* _primaryMesh;
  LineSegmentMeshInstance(
      filament::Engine* engine,
      utils::Entity entity,
      LineSegmentMesh* primaryMesh);
  utils::Entity _entity;
};

//------------------------------------------------------------------------------------------------
// WIREFRAME MESH
//------------------------------------------------------------------------------------------------

class WireframeMeshInstance;

class WireframeMesh : public SceneObject, public IInstanceable {
 public:
  // A triangle surface drawn as a one-sided wireframe. It is built as two primitives that share
  // the geometry: primitive 0 is a "surface" pass that writes depth so the wireframe self-occludes
  // (back faces / far surfaces hide the edges behind them), and primitive 1 draws the antialiased
  // edges, depth-tested against that surface.
  //
  // The caller chooses between two use cases via `surfaceMaterial` and `castShadows`:
  //   1. Pure wireframe (edges only): pass a depth-only material
  //      (@ref ResourceManager::CreateWireframeDepthMaterial) and castShadows=false. The surface
  //      pass contributes depth but no color, so only the edges are visible.
  //   2. Shaded surface + wireframe overlay: pass an opaque surface material
  //      (e.g. @ref ResourceManager::CreateFlatLitOpaqueMaterial) and castShadows=true. The surface
  //      pass shades the model and the edges are drawn on top.
  //
  // `wireframeMaterial` is the edge material (@ref ResourceManager::CreateWireframeMaterial).
  // `normals` provide the surface's per-vertex shading frame and are always required.
  // `isDynamic` allows the geometry to be replaced later via @ref UpdateGeometry.
  static std::unique_ptr<WireframeMesh> CreateWireframeMesh(
      filament::Engine* engine,
      mochi::Span<float const> positions,
      mochi::Span<float const> normals,
      mochi::Span<int const> indices,
      std::shared_ptr<MaterialInstance> wireframeMaterial,
      std::shared_ptr<MaterialInstance> surfaceMaterial,
      bool isClosed = true,
      bool castShadows = false,
      bool isDynamic = false);
  ~WireframeMesh() override;

  void SetColor(filament::math::float4 color);
  bool UpdateGeometry(
      mochi::Span<float const> positions,
      mochi::Span<float const> normals,
      mochi::Span<int const> indices);
  // Lightweight per-frame deform for dynamic meshes: replaces only the position and tangent buffers
  // (topology fixed) without recreating GPU buffers or flushing. `positions`/`normals` are per-node
  // (indexed) data matching the connectivity captured at creation; they are de-indexed into the
  // per-corner vertices internally. Returns false if the mesh is not dynamic.
  bool Update(mochi::Span<float const> positions, mochi::Span<float const> normals);
  void SetMaterial(std::shared_ptr<MaterialInstance> material) override;
  utils::Entity GetRootEntity() const override;
  mochi::Span<utils::Entity const> GetEntities() const override;

  std::unique_ptr<SceneObject> GetInstance() override;
  int GetInstanceCount() const override;
  IInstanceable* GetInstanceable() override;

 private:
  WireframeMesh(
      filament::Engine* engine,
      mochi::Span<float const> positions,
      mochi::Span<float const> normals,
      mochi::Span<int const> indices,
      std::shared_ptr<MaterialInstance> wireframeMaterial,
      std::shared_ptr<MaterialInstance> surfaceMaterial,
      bool isClosed,
      bool castShadows,
      bool isDynamic);

 private:
  friend class WireframeMeshInstance;
  size_t _vertexCount = 0;
  size_t _indexCount = 0;
  utils::Entity _entity;
  filament::VertexBuffer* _vertexBuffer = nullptr;
  filament::IndexBuffer* _indexBuffer = nullptr;
  // Primitive 0 (depth-only pre-pass or flat-lit surface) and primitive 1 (wireframe edges).
  // Owned by the caller via shared_ptr; this mesh only references them.
  std::shared_ptr<MaterialInstance> _surfaceMaterial;
  std::shared_ptr<MaterialInstance> _wireframeMaterial;
  filament::Box _boundingBox;
  // Shadows are enabled only for the combined flat-lit mesh, not a pure wireframe.
  bool _castShadows = false;
  // When dynamic, geometry can be replaced via UpdateGeometry and renderables use DYNAMIC type.
  bool _isDynamic = false;
  // Entities of all live instances, so UpdateGeometry can re-point them at new buffers.
  std::vector<utils::Entity> _instanceEntities;

  // Persistent per-corner (de-indexed) CPU buffers and the per-corner→node connectivity, populated
  // only when dynamic so @ref Update can rewrite the position/tangent slots in place each frame.
  std::vector<filament::math::float3> _dynamicPositions;
  std::vector<filament::math::short4> _dynamicTangents;
  std::vector<float> _dynamicDeindexedNormals;
  std::vector<int> _connectivity;
};

class WireframeMeshInstance : public SceneObject {
 public:
  ~WireframeMeshInstance() override;
  utils::Entity GetRootEntity() const override;
  mochi::Span<utils::Entity const> GetEntities() const override;
  IInstanceable* GetInstanceable() override;

 private:
  friend class WireframeMesh;
  WireframeMesh* _primaryMesh;
  WireframeMeshInstance(filament::Engine* engine, utils::Entity entity, WireframeMesh* primaryMesh);
  utils::Entity _entity;
};

} // namespace mochi_renderer
