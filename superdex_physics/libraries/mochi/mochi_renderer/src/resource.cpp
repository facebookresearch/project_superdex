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

#include <mochi_renderer/windows_compat.h> // Must be first — cleans up Windows macros before Filament headers

#include <mochi_renderer/material.h>
#include <mochi_renderer/mesh.h>
#include <mochi_renderer/resource.h>

#include <filament/IndexBuffer.h>
#include <filament/MaterialInstance.h>
#include <filament/RenderableManager.h>
#include <filament/TransformManager.h>
#include <filament/VertexBuffer.h>

#include <gltfio/FilamentInstance.h>

#include <mochi_core/utils/debug.h>

#include <algorithm>
#include <cctype>

namespace mochi_renderer {

//--------------------------------------------------------------------------------------------------
// RESOURCE
//--------------------------------------------------------------------------------------------------

std::string Resource::GetNameFromPath(mochi::Path const& path) {
  std::string stem = path.GetStem();
  auto const dot = stem.find_last_of('.');
  if (dot != std::string::npos) {
    std::string ext = stem.substr(dot);
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    if (ext == ".mochi") {
      return stem.substr(0, dot);
    }
  }
  return stem;
}

std::string const& Resource::GetName() const {
  return _name;
}

mochi::Path const& Resource::GetPath() const {
  return _path;
}

ResourceType Resource::GetType() const {
  return _type;
}

Resource::Resource(
    filament::Engine* engine,
    std::string const& name,
    mochi::Path const& path,
    ResourceType type)
    : _engine(engine), _name(name), _path(path), _type(type) {}

//--------------------------------------------------------------------------------------------------
// RENDER MODEL
//--------------------------------------------------------------------------------------------------

RenderModel::RenderModel(
    filament::Engine* engine,
    std::string const& name,
    mochi::Path const& path,
    RenderModelFormat originalFormat)
    : Resource(engine, name, path, ResourceType::RenderModel), _originalFormat(originalFormat) {}

RenderModel::~RenderModel() {
  if (_assetLoader && _primaryAsset) {
    _assetLoader->destroyAsset(_primaryAsset);
  }
  if (_ownedVertexBuffer) {
    _engine->destroy(_ownedVertexBuffer);
  }
  if (_ownedIndexBuffer) {
    _engine->destroy(_ownedIndexBuffer);
  }
}

void RenderModel::ConfigureInstance(filament::gltfio::FilamentInstance* instance) {
  // Enable stencil writes on this instance's own material instances. Each FilamentInstance owns
  // its own material instances, so this must run per instance (not just on instance 0).
  size_t const matInstanceCount = instance->getMaterialInstanceCount();
  filament::MaterialInstance* const* const matInstances = instance->getMaterialInstances();
  for (size_t mi = 0; mi < matInstanceCount; ++mi) {
    matInstances[mi]->setStencilWrite(true);
    matInstances[mi]->setStencilOpDepthStencilPass(
        filament::MaterialInstance::StencilOperation::INCR);
  }
  // Valid because RenderModel source data is retained (never releaseSourceData'd).
  instance->recomputeBoundingBoxes();
}

void RenderModel::ApplyGeometryToInstance(filament::gltfio::FilamentInstance* instance) {
  auto& rcm = _engine->getRenderableManager();
  auto const* entities = instance->getEntities();
  size_t const entityCount = instance->getEntityCount();
  for (size_t i = 0; i < entityCount; ++i) {
    auto ri = rcm.getInstance(entities[i]);
    if (!ri.isValid()) {
      continue;
    }
    size_t const primitiveCount = rcm.getPrimitiveCount(ri);
    for (size_t p = 0; p < primitiveCount; ++p) {
      rcm.setGeometryAt(
          ri,
          p,
          filament::RenderableManager::PrimitiveType::TRIANGLES,
          _ownedVertexBuffer,
          _ownedIndexBuffer,
          0,
          _ownedIndexCount);
    }
    rcm.setAxisAlignedBoundingBox(ri, _ownedBounds);
  }
}

filament::gltfio::FilamentInstance* RenderModel::CreateNewInstance() {
  if (static_cast<int>(_instances.size()) >= _maxInstances) {
    MOCHI_LOG_ERROR(
        "RenderModel '%s' reached its instance cap (%d); cannot create more instances.",
        GetName().c_str(),
        _maxInstances);
    return nullptr;
  }
  auto* instance = _assetLoader->createInstance(_primaryAsset);
  if (!instance) {
    MOCHI_LOG_ERROR("Failed to create FilamentInstance for RenderModel '%s'.", GetName().c_str());
    return nullptr;
  }
  ConfigureInstance(instance);
  // If UpdateGeometry() has replaced the model's geometry, re-point this freshly created instance
  // at the current override so it matches the rest of the pool instead of reverting to the
  // original glTF mesh.
  if (_ownedVertexBuffer && _ownedIndexBuffer) {
    ApplyGeometryToInstance(instance);
  }
  _instances.push_back(instance);
  _instanceInUse.push_back(false);
  return instance;
}

void RenderModel::InitializeInitialInstance() {
  // The load path fills _instances with the initial instance(s) via createInstancedAsset but does
  // not touch _instanceInUse; size it to match so slot indices stay in sync from the start. Do
  // this on every path (even when the initial instance is null) so the two vectors never desync.
  _instanceInUse.assign(_instances.size(), false);
  for (auto* instance : _instances) {
    if (instance) {
      ConfigureInstance(instance);
    }
  }
}

std::unique_ptr<SceneObject> RenderModel::GetInstance() {
  // GetInstance()/CreateNewInstance() run on the main engine thread (same assumption as today);
  // a mutex could be added later if cross-thread access is ever required.
  int slot = -1;
  // Only reuse slots below the cap and skip any null instances; a lowered cap must not hand out
  // slots beyond it, and a null slot would crash on dereference below.
  int const scanEnd = std::min(static_cast<int>(_instanceInUse.size()), _maxInstances);
  for (int i = 0; i < scanEnd; ++i) {
    if (!_instanceInUse[i] && _instances[i] != nullptr) {
      slot = i;
      break;
    }
  }
  if (slot < 0) {
    if (!CreateNewInstance()) {
      return nullptr;
    }
    slot = static_cast<int>(_instanceInUse.size()) - 1;
  }
  _instanceInUse[slot] = true;
  auto* instance = _instances[slot];
  auto& tm = _engine->getTransformManager();
  auto ti = tm.getInstance(instance->getRoot());
  tm.setParent(ti, {});
  tm.setTransform(ti, filament::math::mat4f());
  auto& rcm = _engine->getRenderableManager();
  auto const* entities = instance->getEntities();
  size_t const entityCount = instance->getEntityCount();
  for (size_t i = 0; i < entityCount; ++i) {
    auto ri = rcm.getInstance(entities[i]);
    if (!ri.isValid()) {
      continue;
    }
    rcm.setCastShadows(ri, true);
    rcm.setReceiveShadows(ri, true);
    rcm.setLayerMask(ri, 0xFF, 0x01);
  }

  auto ret = std::unique_ptr<RenderModelInstance>(new RenderModelInstance(this, instance, slot));
  ret->SetName(GetName());
  return ret;
}

int RenderModel::GetInstanceCount() const {
  return static_cast<int>(std::count(_instanceInUse.begin(), _instanceInUse.end(), true));
}

int RenderModel::GetMaxInstances() const {
  return _maxInstances;
}

void RenderModel::SetMaxInstances(int max) {
  if (max < 1) {
    MOCHI_LOG_ERROR(
        "RenderModel '%s': SetMaxInstances requires a positive cap (got %d); ignoring.",
        GetName().c_str(),
        max);
    return;
  }
  _maxInstances = max;
}

RenderModelFormat RenderModel::GetOriginalFormat() const {
  return _originalFormat;
}

void RenderModel::UpdateGeometry(
    mochi::Span<float const> positions,
    mochi::Span<float const> normals,
    mochi::Span<int const> indices) {
  if (positions.empty() || indices.empty()) {
    MOCHI_LOG_ERROR("RenderModel::UpdateGeometry called with empty geometry.");
    return;
  }
  MeshBuffers const buffers = CreateModelMeshBuffers(*_engine, positions, normals, indices);
  // Retain the previous override's buffers so they can be freed after the instances are re-pointed
  // at the new geometry below.
  filament::VertexBuffer* const oldVertexBuffer = _ownedVertexBuffer;
  filament::IndexBuffer* const oldIndexBuffer = _ownedIndexBuffer;
  // Adopt the new geometry as the model's owned override before re-pointing instances, so both
  // existing instances (looped below) and any future instances created via CreateNewInstance()
  // share it.
  _ownedVertexBuffer = buffers.vertexBuffer;
  _ownedIndexBuffer = buffers.indexBuffer;
  _ownedIndexCount = indices.size();
  _ownedBounds = buffers.bounds;
  // Re-point every pooled instance (both checked-out and free) at the new geometry, so all
  // live instances update and future GetInstance() calls also yield the updated mesh. The
  // instances share these buffers; per-primitive material instances are left untouched.
  for (auto* instance : _instances) {
    if (!instance) {
      continue;
    }
    ApplyGeometryToInstance(instance);
  }
  // Free the buffers from the previous UpdateGeometry call. On the first call the old
  // geometry is owned by the gltfio asset (freed later by destroyAsset), so nothing to do.
  if (oldVertexBuffer || oldIndexBuffer) {
    // Drain the backend before freeing the old buffers.
    _engine->flushAndWait();
  }
  if (oldVertexBuffer) {
    _engine->destroy(oldVertexBuffer);
  }
  if (oldIndexBuffer) {
    _engine->destroy(oldIndexBuffer);
  }
}

RenderModelInstance::RenderModelInstance(
    RenderModel* model,
    filament::gltfio::FilamentInstance* instance,
    int instanceIndex)
    : SceneObject(model->_engine),
      _model(model),
      _instance(instance),
      _instanceIndex(instanceIndex) {}

RenderModelInstance::~RenderModelInstance() {
  // If a custom material was applied via SetMaterial(), restore the original
  // GLTF material instances so the recycled FilamentInstance doesn't have
  // dangling material pointers after _material is destroyed.
  if (_material || !_originalMaterials.empty()) {
    auto& rcm = _model->_engine->getRenderableManager();
    size_t matIndex = 0;
    auto entities = GetEntities();
    for (size_t i = 0; i < entities.size() && matIndex < _originalMaterials.size(); ++i) {
      auto ri = rcm.getInstance(entities[i]);
      if (!ri.isValid()) {
        continue;
      }
      size_t const primitiveCount = rcm.getPrimitiveCount(ri);
      for (size_t p = 0; p < primitiveCount && matIndex < _originalMaterials.size(); ++p) {
        rcm.setMaterialInstanceAt(ri, p, _originalMaterials[matIndex++]);
      }
    }
  }
  if (_instanceIndex >= 0) {
    _model->_instanceInUse[_instanceIndex] = false;
  }
}

utils::Entity RenderModelInstance::GetRootEntity() const {
  return _instance->getRoot();
}

mochi::Span<utils::Entity const> RenderModelInstance::GetEntities() const {
  return {_instance->getEntities(), _instance->getEntityCount()};
}

void RenderModelInstance::SetMaterial(std::shared_ptr<MaterialInstance> material) {
  // Lazily snapshot original materials on first call.
  if (_originalMaterials.empty()) {
    auto& rcm = _engine->getRenderableManager();
    auto entities = GetEntities();
    for (auto entity : entities) {
      auto ri = rcm.getInstance(entity);
      if (!ri.isValid()) {
        continue;
      }
      size_t const primitiveCount = rcm.getPrimitiveCount(ri);
      for (size_t p = 0; p < primitiveCount; ++p) {
        _originalMaterials.push_back(rcm.getMaterialInstanceAt(ri, p));
      }
    }
  }
  SceneObject::SetMaterial(material);
}

IInstanceable* RenderModelInstance::GetInstanceable() {
  return _model;
}

} // namespace mochi_renderer
