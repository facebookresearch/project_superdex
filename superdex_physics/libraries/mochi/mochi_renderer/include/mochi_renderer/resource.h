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

#include <mochi_renderer/path.h>
#include <mochi_renderer/scene_object.h>

#include <filament/Box.h>
#include <filament/Engine.h>
#include <gltfio/AssetLoader.h>

#include <mochi_core/geometry/model_data.h>
#include <mochi_core/geometry/model_utils.h>

#include <algorithm>
#include <vector>

namespace mochi_renderer {

class ResourceManager;

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

  // Soft cap on the number of concurrently allocated instances for this model.
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

 private:
  friend class ResourceManager;
  RenderModel(
      filament::Engine* engine,
      std::string const& name,
      mochi::Path const& path,
      RenderModelFormat originalFormat);

  // Creates a new FilamentInstance on demand and appends it to the pool. Returns nullptr if the
  // soft cap is reached or gltfio fails. Runs on the main engine thread (same assumption as
  // GetInstance()); a mutex could be added later if cross-thread access is ever required.
  filament::gltfio::FilamentInstance* CreateNewInstance();
  // One-time per-instance setup applied to every instance when it is created: enables stencil
  // write/INCR on the instance's own material instances and recomputes its bounding boxes.
  void ConfigureInstance(filament::gltfio::FilamentInstance* instance);
  // Re-points a single instance at the model's current geometry override (_ownedVertexBuffer/
  // _ownedIndexBuffer). Shared by UpdateGeometry() (existing instances) and CreateNewInstance()
  // (instances grown after an UpdateGeometry() call) so the whole pool stays consistent.
  void ApplyGeometryToInstance(filament::gltfio::FilamentInstance* instance);
  // Applies the one-time setup to the initial instance (index 0) created at load time. Called by
  // ResourceManager after _primaryAsset/_assetLoader are set and resources are loaded.
  void InitializeInitialInstance();

  friend class RenderModelInstance;
  RenderModelFormat _originalFormat;

  filament::gltfio::FilamentAsset* _primaryAsset = nullptr;
  FilamentInstanceVector _instances;
  std::vector<bool> _instanceInUse;
  int _maxInstances = kDefaultMaxInstances;
  filament::gltfio::AssetLoader* _assetLoader = nullptr;
  filament::VertexBuffer* _ownedVertexBuffer = nullptr;
  filament::IndexBuffer* _ownedIndexBuffer = nullptr;
  size_t _ownedIndexCount = 0;
  filament::Box _ownedBounds;
};

class RenderModelInstance : public SceneObject {
 public:
  ~RenderModelInstance() override;
  utils::Entity GetRootEntity() const override;
  mochi::Span<utils::Entity const> GetEntities() const override;
  void SetMaterial(std::shared_ptr<MaterialInstance> material) override;
  IInstanceable* GetInstanceable() override;

 private:
  RenderModelInstance(
      RenderModel* model,
      filament::gltfio::FilamentInstance* instance,
      int instanceIndex);
  friend class RenderModel;
  RenderModel* _model = nullptr;
  filament::gltfio::FilamentInstance* _instance = nullptr;
  int _instanceIndex = -1;
  std::vector<filament::MaterialInstance*> _originalMaterials;
};

} // namespace mochi_renderer
