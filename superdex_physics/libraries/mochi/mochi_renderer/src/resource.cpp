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
#include <mochi_renderer/type_conversions.h>

#include <filament/IndexBuffer.h>
#include <filament/MaterialInstance.h>
#include <filament/RenderableManager.h>
#include <filament/TransformManager.h>
#include <filament/VertexBuffer.h>

#include <gltfio/Animator.h>
#include <gltfio/FilamentAsset.h>
#include <gltfio/FilamentInstance.h>

#include <cgltf.h>

#include <mochi_core/utils/debug.h>

#include <algorithm>
#include <cctype>
#include <limits>
#include <vector>

namespace mochi_renderer {

//--------------------------------------------------------------------------------------------------
// RESOURCE
//--------------------------------------------------------------------------------------------------

std::string Resource::GetNameFromPath(mochi::Path const& path) {
  std::string stem = path.GetStem();
  auto const dot = stem.find_last_of('.');
  if (dot != std::string::npos) {
    std::string ext = stem.substr(dot);
    std::ranges::transform(
        ext, ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
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
  // Destroying the asset destroys its renderables, which may draw the deformable buffers.
  if (_assetLoader && _primaryAsset) {
    _assetLoader->destroyAsset(_primaryAsset);
  }
  _deformation.reset();
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

filament::gltfio::FilamentInstance* RenderModel::CreateConfiguredInstance() {
  // gltfio cannot destroy single instances, so the cap counts every allocated one. Both pools
  // reuse released instances before allocating more, so their sizes stay at their peak use.
  if (static_cast<int>(_instances.size()) + GetDeformableSlotCount() >= _maxInstances) {
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
  return instance;
}

filament::gltfio::FilamentInstance* RenderModel::CreateNewInstance() {
  auto* instance = CreateConfiguredInstance();
  if (!instance) {
    return nullptr;
  }
  // If UpdateGeometry() has replaced the model's geometry, re-point this freshly created instance
  // at the current override so it matches the rest of the pool instead of reverting to the
  // original glTF mesh.
  if (_ownedVertexBuffer && _ownedIndexBuffer) {
    ApplyGeometryToInstance(instance);
  }
  _instances.push_back(instance);
  _instanceInUse.push_back(false);
  _instanceRestBounds.push_back(CaptureInstanceRestBounds(instance));
  return instance;
}

std::vector<filament::Box> RenderModel::CaptureInstanceRestBounds(
    filament::gltfio::FilamentInstance* instance) const {
  std::vector<filament::Box> bounds;
  if (instance == nullptr) {
    return bounds;
  }
  auto& rcm = _engine->getRenderableManager();
  auto const* entities = instance->getEntities();
  size_t const entityCount = instance->getEntityCount();
  bounds.reserve(entityCount);
  for (size_t i = 0; i < entityCount; ++i) {
    auto ri = rcm.getInstance(entities[i]);
    // Keep one entry per entity so indices line up with getEntities() on restore; non-renderable
    // entities get a default box that is never applied.
    bounds.push_back(ri.isValid() ? rcm.getAxisAlignedBoundingBox(ri) : filament::Box{});
  }
  return bounds;
}

void RenderModel::InitializeInitialInstance() {
  // The load path fills _instances with the initial instance(s) via createInstancedAsset but does
  // not touch _instanceInUse; size it to match so slot indices stay in sync from the start. Do
  // this on every path (even when the initial instance is null) so the two vectors never desync.
  _instanceInUse.assign(_instances.size(), false);
  _instanceRestBounds.assign(_instances.size(), {});
  for (size_t i = 0; i < _instances.size(); ++i) {
    if (_instances[i]) {
      ConfigureInstance(_instances[i]);
      _instanceRestBounds[i] = CaptureInstanceRestBounds(_instances[i]);
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
  ResetInstanceState(instance);
  auto ret = CreateInstanceObject(instance, slot);
  ret->SetName(GetName());
  return ret;
}

void RenderModel::ResetInstanceState(filament::gltfio::FilamentInstance* instance) const {
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
}

std::unique_ptr<RenderModelInstance> RenderModel::CreateInstanceObject(
    filament::gltfio::FilamentInstance* instance,
    int instanceIndex) {
  return std::unique_ptr<RenderModelInstance>(
      new RenderModelInstance(this, instance, instanceIndex));
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
  // Released deformable slots keep their buffers until the model is destroyed. Once the geometry is
  // replaced, GetDeformableMesh refuses, so they are never handed out again.
  if (GetDeformableInstanceCount() > 0) {
    MOCHI_LOG_ERROR(
        "RenderModel '%s' has deformable instances checked out; UpdateGeometry cannot replace its "
        "geometry until they are released.",
        GetName().c_str());
    return;
  }
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
  for (size_t i = 0; i < _instances.size(); ++i) {
    if (!_instances[i]) {
      continue;
    }
    ApplyGeometryToInstance(_instances[i]);
    // ApplyGeometryToInstance rebounds the instance to the new geometry, so the captured authored
    // bounds are now stale; refresh them or checkout would restore the previous geometry's box.
    if (i < _instanceRestBounds.size()) {
      _instanceRestBounds[i] = CaptureInstanceRestBounds(_instances[i]);
    }
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
  // Leave the pooled instance in its authored state. SkinnedModelInstance::SetBoneMatrices
  // rewrites each renderable's AABB to the posed box; without this the next user of the recycled
  // instance would inherit that box and treat it as the model's rest bounds.
  if (_instanceIndex >= 0 &&
      static_cast<size_t>(_instanceIndex) < _model->_instanceRestBounds.size()) {
    auto& rcm = _model->_engine->getRenderableManager();
    auto const& restBounds = _model->_instanceRestBounds[_instanceIndex];
    auto const entities = GetEntities();
    for (size_t i = 0; i < entities.size() && i < restBounds.size(); ++i) {
      auto ri = rcm.getInstance(entities[i]);
      if (ri.isValid()) {
        rcm.setAxisAlignedBoundingBox(ri, restBounds[i]);
      }
    }
  }
  if (_instanceIndex >= 0) {
    _model->_instanceInUse[_instanceIndex] = false;
  }
}

filament::Box RenderModelInstance::GetAuthoredEntityBounds(size_t entityIndex) const {
  if (_model == nullptr || _instanceIndex < 0) {
    return {};
  }
  auto const& perInstance = _model->_instanceRestBounds;
  if (static_cast<size_t>(_instanceIndex) >= perInstance.size() ||
      entityIndex >= perInstance[_instanceIndex].size()) {
    return {};
  }
  return perInstance[_instanceIndex][entityIndex];
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

//--------------------------------------------------------------------------------------------------
// SKINNED MODEL
//--------------------------------------------------------------------------------------------------

SkinnedModel::SkinnedModel(
    filament::Engine* engine,
    std::string const& name,
    mochi::Path const& path,
    RenderModelFormat originalFormat)
    : RenderModel(engine, name, path, originalFormat) {}

std::unique_ptr<RenderModelInstance> SkinnedModel::CreateInstanceObject(
    filament::gltfio::FilamentInstance* instance,
    int instanceIndex) {
  return std::unique_ptr<RenderModelInstance>(
      new SkinnedModelInstance(this, instance, instanceIndex));
}

SkinnedModelInstance::SkinnedModelInstance(
    RenderModel* model,
    filament::gltfio::FilamentInstance* instance,
    int instanceIndex)
    : RenderModelInstance(model, instance, instanceIndex) {
  // Eager rather than lazy: a skin that is only ever displayed (a thumbnail, the model viewer)
  // never calls SetBoneMatrices, and would otherwise keep whatever bounds gltfio produced at load.
  EnsureJointCache();
}

void SkinnedModelInstance::EnsureJointCache() {
  if (_jointCacheReady) {
    return;
  }
  _jointCacheReady = true;
  auto* instance = GetFilamentInstance();
  if (instance->getSkinCount() > 0) {
    size_t const count = instance->getJointCountAt(0);
    utils::Entity const* joints = instance->getJointsAt(0);
    _joints.assign(joints, joints + count);
  }
  // Snapshot each renderable's load-time (rest) AABB while the renderables still carry the GLB's
  // authored bounds (this runs before the first pose is applied). SetBoneMatrices rebuilds a posed
  // cull box from these each frame. A skin GLB's renderables are all driven by skin 0; any
  // incidental non-skinned renderable would just get a conservatively larger box, which is safe for
  // culling.
  auto& rm = _engine->getRenderableManager();
  {
    auto const entities = GetEntities();
    for (size_t i = 0; i < entities.size(); ++i) {
      auto ri = rm.getInstance(entities[i]);
      if (!ri.isValid()) {
        continue;
      }
      _skinnedRenderables.push_back(entities[i]);
      _skinnedEntityIndices.push_back(i);
    }
  }

  // Per-joint rest sub-AABBs (skin 0), in the GLB's model space, from the retained cgltf source:
  // for each vertex, expand the sub-box of every joint that influences it (weight > 0). JOINTS_0
  // indices are in skin-0 joint order, matching getJointsAt(0) / getInverseBindMatricesAt(0).
  // SetBoneMatrices transforms each joint's own sub-box for a tight posed AABB; if the source is
  // unavailable this stays empty and it falls back to the whole rest box.
  if (!_joints.empty()) {
    auto const* asset = instance->getAsset();
    auto const* src = asset != nullptr
        ? static_cast<cgltf_data const*>(
              const_cast<filament::gltfio::FilamentAsset*>(asset)->getSourceAsset())
        : nullptr;
    if (src != nullptr && src->skins_count > 0) {
      size_t const jointCount = _joints.size();
      std::vector<filament::math::float3> boneMin(
          jointCount, filament::math::float3{std::numeric_limits<float>::max()});
      std::vector<filament::math::float3> boneMax(
          jointCount, filament::math::float3{std::numeric_limits<float>::lowest()});
      cgltf_skin const* const baseSkin = &src->skins[0];
      std::vector<float> pos;
      std::vector<float> jnt;
      std::vector<float> wgt;
      for (size_t n = 0; n < src->nodes_count; ++n) {
        cgltf_node const& node = src->nodes[n];
        if (node.mesh == nullptr || node.skin != baseSkin) {
          continue;
        }
        for (size_t pi = 0; pi < node.mesh->primitives_count; ++pi) {
          cgltf_primitive const& prim = node.mesh->primitives[pi];
          cgltf_accessor const* posA = nullptr;
          cgltf_accessor const* jointA = nullptr;
          cgltf_accessor const* weightA = nullptr;
          for (size_t as = 0; as < prim.attributes_count; ++as) {
            switch (prim.attributes[as].type) {
              case cgltf_attribute_type_position:
                posA = prim.attributes[as].data;
                break;
              case cgltf_attribute_type_joints:
                jointA = prim.attributes[as].data;
                break;
              case cgltf_attribute_type_weights:
                weightA = prim.attributes[as].data;
                break;
              default:
                break;
            }
          }
          if (posA == nullptr || jointA == nullptr || weightA == nullptr) {
            continue;
          }
          size_t const vcount = posA->count;
          pos.resize(vcount * 3);
          jnt.resize(vcount * 4);
          wgt.resize(vcount * 4);
          cgltf_accessor_unpack_floats(posA, pos.data(), pos.size());
          cgltf_accessor_unpack_floats(jointA, jnt.data(), jnt.size());
          cgltf_accessor_unpack_floats(weightA, wgt.data(), wgt.size());
          for (size_t i = 0; i < vcount; ++i) {
            filament::math::float3 const p{pos[i * 3 + 0], pos[i * 3 + 1], pos[i * 3 + 2]};
            for (int k = 0; k < 4; ++k) {
              float const w = wgt[i * 4 + k];
              int const bone = static_cast<int>(jnt[i * 4 + k]);
              if (w <= 0.0f || bone < 0 || static_cast<size_t>(bone) >= jointCount) {
                continue;
              }
              boneMin[bone] = min(boneMin[bone], p);
              boneMax[bone] = max(boneMax[bone], p);
            }
          }
        }
      }
      // Rest bone matrices straight from the source: jointWorld_rest * inverseBind. Both come
      // from cgltf, so this does not depend on filament's transform state having been propagated.
      if (baseSkin->inverse_bind_matrices != nullptr &&
          baseSkin->joints_count >= static_cast<cgltf_size>(jointCount)) {
        std::vector<float> ibm(baseSkin->joints_count * 16);
        cgltf_accessor_unpack_floats(baseSkin->inverse_bind_matrices, ibm.data(), ibm.size());
        _restBoneMatrices.assign(jointCount, filament::math::mat4f{});
        for (size_t j = 0; j < jointCount; ++j) {
          float world[16];
          cgltf_node_transform_world(baseSkin->joints[j], world);
          filament::math::mat4f const jointWorld =
              *reinterpret_cast<filament::math::mat4f const*>(world);
          filament::math::mat4f const inverseBind =
              *reinterpret_cast<filament::math::mat4f const*>(&ibm[j * 16]);
          _restBoneMatrices[j] = jointWorld * inverseBind;
        }
      }
      _boneRestBoxes.assign(jointCount, filament::Box{{0.0f, 0.0f, 0.0f}, {-1.0f, -1.0f, -1.0f}});
      for (size_t b = 0; b < jointCount; ++b) {
        if (boneMin[b].x <= boneMax[b].x) {
          _boneRestBoxes[b] =
              filament::Box{(boneMin[b] + boneMax[b]) * 0.5f, (boneMax[b] - boneMin[b]) * 0.5f};
        }
      }
    }
  }

  // Establish the rest-pose cull box now, from source data. gltfio's own recomputeBoundingBoxes()
  // needs the joint world transforms to have been propagated, which is not guaranteed this early --
  // a skin drawn before that (an asset thumbnail) would otherwise be framed by bind-space bounds
  // while rendering at its rest pose.
  if (!_restBoneMatrices.empty()) {
    // Upload the rest pose as well as its bounds. Instances are pooled, so this one may still hold
    // the previous user's bone matrices -- a skin posed onto an arm, say -- which would draw the
    // mesh far from the rest bounds the camera frames. _jointCacheReady is already set, so the
    // EnsureJointCache() call at the top of SetBoneMatrices returns immediately.
    SetBoneMatrices(mochi::MakeConstSpan(_restBoneMatrices));
  }
}

void SkinnedModelInstance::SetBoneMatrices(mochi::Span<filament::math::mat4f const> boneMatrices) {
  EnsureJointCache();
  if (_joints.empty() || _skinnedRenderables.empty() || boneMatrices.empty()) {
    return;
  }
  auto& rm = _engine->getRenderableManager();
  size_t const count = std::min(boneMatrices.size(), _joints.size());
  for (utils::Entity const entity : _skinnedRenderables) {
    auto ri = rm.getInstance(entity);
    if (ri.isValid()) {
      rm.setBones(ri, boneMatrices.data(), count, 0);
    }
  }
  ApplyPosedBounds(boneMatrices);
}

void SkinnedModelInstance::ApplyPosedBounds(mochi::Span<filament::math::mat4f const> boneMatrices) {
  if (_skinnedRenderables.empty() || boneMatrices.empty()) {
    return;
  }
  auto& rm = _engine->getRenderableManager();
  size_t const count = std::min(boneMatrices.size(), _joints.size());

  // Rebuild every renderable's cull box for this pose: skinning does not update the load-time
  // bounds, so the skin would drift outside them and be frustum-culled.
  //
  // The box is built PER RENDERABLE. _boneRestBoxes spans every primitive driven by skin 0, so a
  // bone's box is generally bigger than any one renderable; handing that union to each renderable
  // would give them all bounds covering the whole skin, which inflates whatever reads renderable
  // bounds (the scene floor takes the lowest AABB point). Clipping each bone box to the
  // renderable's own rest bounds fixes that and stays correct: a vertex of renderable r weighted
  // to bone j lies in both boxes, so it survives the clip, and every skinned vertex is a convex
  // combination of {bone_j * v}, which the union of the transformed clipped boxes bounds. A bone
  // whose box misses the renderable entirely contributes nothing. With a single skinned renderable
  // the clip is a no-op.
  bool const haveSubBoxes = _boneRestBoxes.size() == _joints.size();
  for (size_t r = 0; r < _skinnedRenderables.size(); ++r) {
    auto ri = rm.getInstance(_skinnedRenderables[r]);
    if (!ri.isValid()) {
      continue;
    }
    filament::Box const rest = GetAuthoredEntityBounds(_skinnedEntityIndices[r]);
    filament::math::float3 const restMin = rest.center - rest.halfExtent;
    filament::math::float3 const restMax = rest.center + rest.halfExtent;
    filament::math::float3 minPt{
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max()};
    filament::math::float3 maxPt{
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest()};
    bool anyValid = false;
    auto accumulate = [&](filament::math::mat4f const& bone,
                          filament::math::float3 const& boxMin,
                          filament::math::float3 const& boxMax) {
      for (int corner = 0; corner < 8; ++corner) {
        filament::math::float4 const p{
            (corner & 1) ? boxMax.x : boxMin.x,
            (corner & 2) ? boxMax.y : boxMin.y,
            (corner & 4) ? boxMax.z : boxMin.z,
            1.0f};
        filament::math::float4 const tp = bone * p;
        minPt.x = std::min(minPt.x, tp.x);
        minPt.y = std::min(minPt.y, tp.y);
        minPt.z = std::min(minPt.z, tp.z);
        maxPt.x = std::max(maxPt.x, tp.x);
        maxPt.y = std::max(maxPt.y, tp.y);
        maxPt.z = std::max(maxPt.z, tp.z);
      }
      anyValid = true;
    };

    if (haveSubBoxes) {
      for (size_t j = 0; j < count; ++j) {
        filament::Box const& bone = _boneRestBoxes[j];
        if (bone.halfExtent.x < 0.0f) {
          continue; // this bone influences no vertices
        }
        filament::math::float3 const clipMin = max(bone.center - bone.halfExtent, restMin);
        filament::math::float3 const clipMax = min(bone.center + bone.halfExtent, restMax);
        if (clipMin.x > clipMax.x || clipMin.y > clipMax.y || clipMin.z > clipMax.z) {
          continue; // this bone drives no vertex of this renderable
        }
        accumulate(boneMatrices[j], clipMin, clipMax);
      }
    }
    if (!anyValid) {
      // No usable sub-boxes: transform the whole rest box by every bone (conservative).
      for (size_t j = 0; j < count; ++j) {
        accumulate(boneMatrices[j], restMin, restMax);
      }
    }
    if (anyValid) {
      rm.setAxisAlignedBoundingBox(
          ri, filament::Box{(minPt + maxPt) * 0.5f, (maxPt - minPt) * 0.5f});
    }
  }
}

} // namespace mochi_renderer
