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

#include <mochi_renderer/deformable_mesh.h>
#include <mochi_renderer/path.h>
#include <mochi_renderer/scene_object.h>

#include <filament/Box.h>
#include <filament/Engine.h>
#include <gltfio/AssetLoader.h>

#include <mochi_core/geometry/model_data.h>
#include <mochi_core/geometry/model_utils.h>
#include <mochi_core/utils/transform_rt.h>

#include <algorithm>
#include <optional>
#include <vector>

namespace mochi {
class Error;
} // namespace mochi

namespace mochi_renderer {

class ResourceManager;
class DeformableModelInstance;

//--------------------------------------------------------------------------------------------------
// INSTANCEABLE
//--------------------------------------------------------------------------------------------------

class IInstanceable {
 public:
  virtual ~IInstanceable() = default;
  virtual std::unique_ptr<SceneObject> GetInstance() = 0;
  virtual int GetInstanceCount() const = 0;
};

//--------------------------------------------------------------------------------------------------
// RESOURCE
//--------------------------------------------------------------------------------------------------

enum class ResourceType { RenderModel, Ibl, Invalid };

class Resource {
 public:
  virtual ~Resource() = default;
  std::string const& GetName() const;
  mochi::Path const& GetPath() const;
  ResourceType GetType() const;

  static std::string GetNameFromPath(mochi::Path const& path);

 protected:
  friend class ResourceManager;
  Resource(
      filament::Engine* engine,
      std::string const& name,
      mochi::Path const& path,
      ResourceType type);
  void SetPath(mochi::Path const& path) {
    _path = path;
  }
  void SetName(std::string const& name) {
    _name = name;
  }
  filament::Engine* _engine = nullptr;
  std::string _name;
  mochi::Path _path;
  ResourceType _type = ResourceType::Invalid;
};

//--------------------------------------------------------------------------------------------------
// RENDER MODEL
//--------------------------------------------------------------------------------------------------

// The original disk format of a render model (for logging/info only).
enum class RenderModelFormat { Gltf, Stl, Obj, Collada, MochiModel, Cad };

class RenderModelInstance;

// Pool of glTF instances backing a RenderModel; populated by the ResourceManager load paths.
using FilamentInstanceVector = std::vector<filament::gltfio::FilamentInstance*>;

class RenderModel : public Resource, public IInstanceable {
 public:
  // Default soft cap on concurrently allocated instances per model. Instances grow lazily on
  // demand up to this cap; exceeding it logs an error and returns nullptr, catching runaway
  // allocation bugs. Override per model with SetMaxInstances().
  static constexpr int kDefaultMaxInstances = 1024;
  ~RenderModel() override;

  std::unique_ptr<SceneObject> GetInstance() override;
  int GetInstanceCount() const override;
  RenderModelFormat GetOriginalFormat() const;

  // Soft cap on the number of concurrently allocated instances for this model, including
  // deformable ones.
  int GetMaxInstances() const;
  void SetMaxInstances(int max);

  // Replaces the geometry of every pooled instance with the supplied mesh, so all live
  // instances across all scenes/editors update at once. The instances share a single
  // VertexBuffer/IndexBuffer owned by this model; per-instance material instances are
  // preserved. Supports changed vertex/index counts.
  void UpdateGeometry(
      mochi::Span<float const> positions,
      mochi::Span<float const> normals,
      mochi::Span<int const> indices);

  // The rest mesh accepted by this model's independently deformable instances, in the model root's
  // local frame. Only a model loaded from a binary GLB file that embeds its buffers is deformable,
  // and only when the GLB has exactly one node with a mesh, whose one primitive is a triangle list
  // without skinning, morph targets, GPU instancing, material variants, animation of the node or
  // its ancestors, or texture-coordinate sets other than TEXCOORD_0. The first call reads and
  // decodes the file again, which must still hold the bytes that were loaded. Fails after
  // UpdateGeometry has replaced the model's geometry. The spans remain valid until the model is
  // unloaded.
  std::optional<DeformableMeshView> GetDeformableMesh(mochi::Error& error);
  // An instance whose vertices the caller sets, drawn with this model's materials and transforms.
  // Fails when GetDeformableMesh does, or when no instance is available (see GetMaxInstances).
  // While any is checked out, UpdateGeometry is refused.
  std::unique_ptr<DeformableModelInstance> GetDeformableInstance(mochi::Error& error);

 protected:
  friend class ResourceManager;
  RenderModel(
      filament::Engine* engine,
      std::string const& name,
      mochi::Path const& path,
      RenderModelFormat originalFormat);

  // Factory for the SceneObject handed out by GetInstance(). The base returns a plain
  // RenderModelInstance; SkinnedModel overrides it to return a SkinnedModelInstance. Runs on the
  // main engine thread (same assumption as GetInstance()).
  virtual std::unique_ptr<RenderModelInstance> CreateInstanceObject(
      filament::gltfio::FilamentInstance* instance,
      int instanceIndex);

 private:
  class DeformationState;
  struct DeformationStateDeleter {
    void operator()(DeformationState* state) const;
  };

  // Creates a new FilamentInstance on demand and appends it to the pool. Returns nullptr if the
  // soft cap is reached or gltfio fails. Runs on the main engine thread (same assumption as
  // GetInstance()); a mutex could be added later if cross-thread access is ever required.
  filament::gltfio::FilamentInstance* CreateNewInstance();
  // Creates and configures a FilamentInstance outside the pool, logging why when the soft cap is
  // reached or gltfio fails.
  filament::gltfio::FilamentInstance* CreateConfiguredInstance();
  // Returns a checked-out instance to its default placement, shadowing and layer mask.
  void ResetInstanceState(filament::gltfio::FilamentInstance* instance) const;
  // One-time per-instance setup applied to every instance when it is created: enables stencil
  // write/INCR on the instance's own material instances and recomputes its bounding boxes.
  // Snapshot every entity's current AABB, in getEntities() order, as this instance's rest bounds.
  std::vector<filament::Box> CaptureInstanceRestBounds(
      filament::gltfio::FilamentInstance* instance) const;
  void ConfigureInstance(filament::gltfio::FilamentInstance* instance);
  // Re-points a single instance at the model's current geometry override (_ownedVertexBuffer/
  // _ownedIndexBuffer). Shared by UpdateGeometry() (existing instances) and CreateNewInstance()
  // (instances grown after an UpdateGeometry() call) so the whole pool stays consistent.
  void ApplyGeometryToInstance(filament::gltfio::FilamentInstance* instance);
  // Applies the one-time setup to the initial instance (index 0) created at load time. Called by
  // ResourceManager after _primaryAsset/_assetLoader are set and resources are loaded.
  void InitializeInitialInstance();
  void ReleaseDeformableInstance(int slot);
  void SetDeformableVertices(
      int slot,
      mochi::Span<float const> positions,
      mochi::Span<float const> normals);
  void CopyDeformableVertices(int from, int to);
  int GetDeformableVertexCount() const;
  int GetDeformableInstanceCount() const;
  int GetDeformableSlotCount() const;

  friend class RenderModelInstance;
  friend class DeformableModelInstance;
  RenderModelFormat _originalFormat;

  filament::gltfio::FilamentAsset* _primaryAsset = nullptr;
  FilamentInstanceVector _instances;
  std::vector<bool> _instanceInUse;
  // Authored renderable bounds per pooled instance, one entry per entity (in getEntities() order),
  // captured before anything can overwrite them and restored on checkout. Instances are pooled and
  // reused, and SkinnedModelInstance::SetBoneMatrices rewrites a renderable's AABB to the posed
  // box; without this, the next user of a recycled instance would read that stale, inflated box as
  // if it were the model's rest bounds -- and each reuse would inflate it further.
  std::vector<std::vector<filament::Box>> _instanceRestBounds;
  int _maxInstances = kDefaultMaxInstances;
  filament::gltfio::AssetLoader* _assetLoader = nullptr;
  filament::VertexBuffer* _ownedVertexBuffer = nullptr;
  filament::IndexBuffer* _ownedIndexBuffer = nullptr;
  size_t _ownedIndexCount = 0;
  filament::Box _ownedBounds;
  // Hash of the loaded GLB file's bytes; empty for a model not loaded from a glTF file.
  std::optional<size_t> _glbHash;
  // Created by the first GetDeformableMesh call and kept, with every slot it allocates, until the
  // model is destroyed, so the rest mesh spans stay valid.
  std::unique_ptr<DeformationState, DeformationStateDeleter> _deformation;
};

class RenderModelInstance : public SceneObject {
 public:
  ~RenderModelInstance() override;
  utils::Entity GetRootEntity() const override;
  mochi::Span<utils::Entity const> GetEntities() const override;
  void SetMaterial(std::shared_ptr<MaterialInstance> material) override;
  IInstanceable* GetInstanceable() override;

 protected:
  RenderModelInstance(
      RenderModel* model,
      filament::gltfio::FilamentInstance* instance,
      int instanceIndex);
  // Authored (pre-pose) bounds of this instance's entity at @p entityIndex, in GetEntities()
  // order. Returns an empty box when unavailable. Subclasses that overwrite a renderable's AABB
  // (SkinnedModelInstance) read the untouched bounds through this rather than caching their own.
  filament::Box GetAuthoredEntityBounds(size_t entityIndex) const;
  // Accessor for subclasses (SkinnedModelInstance) that need the backing gltfio instance to reach
  // its skin joints.
  filament::gltfio::FilamentInstance* GetFilamentInstance() const {
    return _instance;
  }
  RenderModel* GetRenderModel() const {
    return _model;
  }

 private:
  friend class RenderModel;
  RenderModel* _model = nullptr;
  filament::gltfio::FilamentInstance* _instance = nullptr;
  int _instanceIndex = -1;
  std::vector<filament::MaterialInstance*> _originalMaterials;
};

//--------------------------------------------------------------------------------------------------
// SKINNED MODEL
//--------------------------------------------------------------------------------------------------

// A glTF/GLB render model that carries a skeleton + per-vertex skin (JOINTS_0/WEIGHTS_0) and is
// deformed on the GPU by driving its joint transforms each frame. Shares all loading, pooling, and
// instancing machinery with RenderModel; the only difference is that GetInstance() hands out a
// SkinnedModelInstance whose joints can be posed. Produced by the ResourceManager when a loaded GLB
// contains a skin (see LoadGltf); the geometry, skeleton, and inverse-bind matrices come straight
// from the GLB (gltfio builds them), so the model root and joint transforms are expressed in the
// GLB's authored frame.
class SkinnedModel : public RenderModel {
 protected:
  friend class ResourceManager;
  SkinnedModel(
      filament::Engine* engine,
      std::string const& name,
      mochi::Path const& path,
      RenderModelFormat originalFormat);

  std::unique_ptr<RenderModelInstance> CreateInstanceObject(
      filament::gltfio::FilamentInstance* instance,
      int instanceIndex) override;
};

class SkinnedModelInstance : public RenderModelInstance {
 public:
  // Drives the skin by per-bone GPU skinning matrices (renderable-local, root-relative):
  // boneMatrix_j = currentLinkWorldRel_j * inverse(restLinkWorldRel_j), built from the
  // ARTICULATION's own current + rest link frames. The GLB's baked skeleton and inverse-bind
  // matrices are never used, so a render GLB authored in a different bind convention (e.g. a
  // DCC/Unreal export whose joint frames differ from the articulation) still deforms correctly.
  // Mirrors WireframeMesh::SetBoneMatrices; boneMatrix_j applies to vertices weighted to skin
  // joint j. The model root itself is placed via the usual SceneObject transform.
  void SetBoneMatrices(mochi::Span<filament::math::mat4f const> boneMatrices);

  SkinnedModelInstance* AsSkinnedModelInstance() override {
    return this;
  }

 private:
  friend class SkinnedModel;
  SkinnedModelInstance(
      RenderModel* model,
      filament::gltfio::FilamentInstance* instance,
      int instanceIndex);
  void EnsureJointCache();
  // Apply a posed cull box to each skinned renderable for @p boneMatrices. Shared by the rest-pose
  // bounds established at construction and by SetBoneMatrices.
  void ApplyPosedBounds(mochi::Span<filament::math::mat4f const> boneMatrices);
  // Bone matrices for the GLB's own rest pose (jointWorld_rest * inverseBind), derived entirely
  // from the retained cgltf source. Filament's recomputeBoundingBoxes() needs the joint world
  // transforms to be settled, which is not guaranteed at load, so rest bounds are computed from
  // source data instead and are the same whenever they are asked for.
  std::vector<filament::math::mat4f> _restBoneMatrices;

  std::vector<utils::Entity> _joints; // skin 0 joints, in skin.joints (== link) order
  bool _jointCacheReady = false;
  // Renderables skinned by skin 0, paired (by index) with their load-time (rest) object-space AABBs
  // captured before any pose is applied. Used by SetBoneMatrices to rebuild a posed cull box.
  std::vector<utils::Entity> _skinnedRenderables;
  // Position of each entry of _skinnedRenderables within GetEntities(), used to look up that
  // renderable's authored bounds from the model. Storing the index rather than a copy of the box
  // keeps one source of truth, so a later UpdateGeometry is picked up automatically.
  std::vector<size_t> _skinnedEntityIndices;
  // Per-bone (skin 0) rest-pose sub-AABB in the GLB's model space: the bounds of just the vertices
  // each joint influences (weight > 0), computed once from the retained cgltf source.
  // SetBoneMatrices transforms each joint's OWN sub-box (not the whole mesh box) so the posed AABB
  // stays tight -- a whole-box-per-joint union over-bounds badly and drags the scene floor far
  // below the skin. An unused joint's box is invalid (halfExtent.x < 0). Empty if the cgltf source
  // was unavailable, in which case SetBoneMatrices falls back to the whole rest box.
  std::vector<filament::Box> _boneRestBoxes;
};

} // namespace mochi_renderer
