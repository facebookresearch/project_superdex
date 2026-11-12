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

#include <mochi_renderer/deformable_model_instance.h>
#include <mochi_renderer/utils.h>

#include "deformable_source.h"
#include "gltf_asset_path.h"
#include "mesh_buffers.h"

#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/error.h>
#include <mochi_core/utils/log.h>

#include <filament/IndexBuffer.h>
#include <filament/Material.h>
#include <filament/MaterialInstance.h>
#include <filament/RenderableManager.h>
#include <filament/VertexBuffer.h>
#include <geometry/SurfaceOrientation.h>
#include <gltfio/FilamentInstance.h>
#include <math/mat3.h>
#include <math/mat4.h>
#include <math/vec2.h>
#include <math/vec4.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <utility>
#include <vector>

namespace mochi_renderer {

namespace {

using filament::math::float2;
using filament::math::float3;
using filament::math::float4;
using filament::math::mat3f;
using filament::math::mat4f;
using filament::math::short4;
using filament::math::uint3;

bool IsFinite(mat4f const& matrix) {
  for (size_t column = 0; column < 4; ++column) {
    for (size_t row = 0; row < 4; ++row) {
      if (!std::isfinite(matrix[column][row])) {
        return false;
      }
    }
  }
  return true;
}

float3 TransformPoint(mat4f const& transform, float3 const& point) {
  return (transform * float4(point, 1.0f)).xyz;
}

// Uploads an owned copy of @p values to @p slot of @p vertexBuffer.
template <typename T>
void UploadCopy(
    filament::Engine& engine,
    filament::VertexBuffer& vertexBuffer,
    int slot,
    std::vector<T> const& values) {
  auto* copy = new T[values.size()];
  std::copy(values.begin(), values.end(), copy);
  vertexBuffer.setBufferAt(
      engine,
      static_cast<uint8_t>(slot),
      filament::VertexBuffer::BufferDescriptor(
          copy, values.size() * sizeof(T), [](void* buffer, size_t, void*) {
            delete[] static_cast<T*>(buffer);
          }));
}

// The number of renderables among @p instance's entities, and in @p entity the last one found.
size_t FindRenderable(
    filament::RenderableManager const& rcm,
    filament::gltfio::FilamentInstance const& instance,
    utils::Entity& entity) {
  size_t count = 0;
  for (size_t i = 0; i < instance.getEntityCount(); ++i) {
    if (rcm.getInstance(instance.getEntities()[i]).isValid()) {
      entity = instance.getEntities()[i];
      ++count;
    }
  }
  return count;
}

} // namespace

class RenderModel::DeformationState {
 public:
  struct Slot {
    filament::gltfio::FilamentInstance* instance = nullptr;
    utils::Entity entity;
    filament::VertexBuffer* vertexBuffer = nullptr;
    std::vector<float3> positions;
    std::vector<short4> tangents;
    bool inUse = false;
  };

  DeformationState(RenderModel& model, DeformableSourceData source)
      : _model(model), _source(std::move(source)) {
    Initialize();
  }

  DeformationState(DeformationState const&) = delete;
  DeformationState& operator=(DeformationState const&) = delete;
  DeformationState(DeformationState&&) = delete;
  DeformationState& operator=(DeformationState&&) = delete;

  ~DeformationState() {
    for (Slot const& slot : _slots) {
      GetEngine().destroy(slot.vertexBuffer);
    }
    if (_indexBuffer != nullptr) {
      GetEngine().destroy(_indexBuffer);
    }
  }

  // Why the model cannot be deformed, as a string literal; null when it can.
  char const* GetError() const {
    return _source.error;
  }

  DeformableMeshView GetMesh() const {
    return {
        .restPositions = mochi::MakeConstSpan(_restPositions),
        .triangleIndices = mochi::MakeConstSpan(_source.indices),
    };
  }

  std::unique_ptr<DeformableModelInstance> Checkout(mochi::Error& error) {
    auto slot =
        std::ranges::find_if(_slots, [](Slot const& candidate) { return !candidate.inUse; });
    if (slot == _slots.end()) {
      filament::gltfio::FilamentInstance* const instance = _model.CreateConfiguredInstance();
      if (instance == nullptr) {
        MOCHI_ERROR_SET(error, "No more instances of the render model can be created.");
        return nullptr;
      }
      _slots.push_back(PrepareSlot(instance));
      slot = std::prev(_slots.end());
    }
    int const index = static_cast<int>(slot - _slots.begin());
    Reset(index);
    slot->inUse = true;
    return std::unique_ptr<DeformableModelInstance>(
        new DeformableModelInstance(&_model, slot->instance, index));
  }

  void Release(int slot) {
    if (slot >= 0 && static_cast<size_t>(slot) < _slots.size()) {
      _slots[slot].inUse = false;
    }
  }

  int GetInstanceCount() const {
    return static_cast<int>(
        std::count_if(_slots.begin(), _slots.end(), [](Slot const& slot) { return slot.inUse; }));
  }

  int GetSlotCount() const {
    return static_cast<int>(_slots.size());
  }

  int GetVertexCount() const {
    return static_cast<int>(_source.positions.size() / 3);
  }

  void
  SetVertices(int slotIndex, mochi::Span<float const> positions, mochi::Span<float const> normals) {
    size_t const expected = _source.positions.size();
    if (positions.size() != expected || normals.size() != expected) {
      MOCHI_LOG_ERROR(
          "Expected %zu deformable positions and normals, got %zu and %zu.",
          expected,
          positions.size(),
          normals.size());
      return;
    }
    Slot& slot = _slots[slotIndex];
    size_t const vertexCount = expected / 3;
    _localNormals.resize(vertexCount);
    // Normals map by the inverse-transpose of nodeFromRoot, which is the transpose of rootFromNode.
    mat3f const rootFromNodeLinear = _rootFromNode.upperLeft();
    for (size_t i = 0; i < vertexCount; ++i) {
      slot.positions[i] = TransformPoint(
          _nodeFromRoot, float3{positions[3 * i], positions[3 * i + 1], positions[3 * i + 2]});
      _localNormals[i] = normalize(
          transpose(rootFromNodeLinear) *
          float3{normals[3 * i], normals[3 * i + 1], normals[3 * i + 2]});
    }
    UpdateTangents(slot, reinterpret_cast<float const*>(_localNormals.data()));
    Upload(slot);
  }

  void Copy(int from, int to) {
    _slots[to].positions = _slots[from].positions;
    _slots[to].tangents = _slots[from].tangents;
    Upload(_slots[to]);
  }

 private:
  filament::Engine& GetEngine() const {
    return *_model._engine;
  }

  // Validates the model's initial instance and computes the rest mesh, or sets _source.error.
  void Initialize() {
    if (_source.error != nullptr) {
      return;
    }
    size_t const vertexCount = _source.positions.size() / 3;
    if (_source.normals.empty()) {
      ComputeVertexNormalsAreaWeighted(_source.positions, _source.indices, _source.normals);
    }
    _hasTexcoords = !_source.texcoords.empty();
    if (!_hasTexcoords) {
      _source.texcoords.assign(2 * vertexCount, 0.0f);
    }
    if (_source.colors.empty()) {
      _source.colors.assign(4 * vertexCount, 1.0f);
    }

    filament::gltfio::FilamentInstance const* const instance =
        _model._instances.empty() ? nullptr : _model._instances.front();
    if (instance == nullptr || instance->getSkinCount() != 0) {
      _source.error = "The render model's Filament instance is missing or skinned.";
      return;
    }
    auto& rcm = GetEngine().getRenderableManager();
    utils::Entity entity;
    if (FindRenderable(rcm, *instance, entity) != 1) {
      _source.error = "Filament did not create exactly one renderable for the GLB.";
      return;
    }
    auto const ri = rcm.getInstance(entity);
    if (rcm.getPrimitiveCount(ri) != 1 || rcm.getInstanceCount(ri) != 1 ||
        rcm.getMorphTargetCount(ri) != 0) {
      _source.error =
          "Filament's renderable for the GLB is instanced, morphed, or has more than "
          "one primitive.";
      return;
    }
    filament::MaterialInstance const* const material = rcm.getMaterialInstanceAt(ri, 0);
    if (material == nullptr) {
      _source.error = "The GLB's renderable has no material.";
      return;
    }
    filament::AttributeBitset supported;
    for (filament::VertexAttribute const attribute :
         {filament::VertexAttribute::POSITION,
          filament::VertexAttribute::TANGENTS,
          filament::VertexAttribute::UV0,
          filament::VertexAttribute::UV1,
          filament::VertexAttribute::COLOR}) {
      supported.set(attribute);
    }
    if ((material->getMaterial()->getRequiredAttributes() & ~supported).any()) {
      _source.error =
          "The GLB's material requires vertex attributes that a deformable instance "
          "does not provide.";
      return;
    }
    static_assert(sizeof(_rootFromNode) == sizeof(_source.rootFromNode));
    std::memcpy(&_rootFromNode, _source.rootFromNode.data(), sizeof(_rootFromNode));
    _nodeFromRoot = inverse(_rootFromNode);
    if (!IsFinite(_rootFromNode) || !IsFinite(_nodeFromRoot)) {
      _source.error = "The GLB's mesh node has a singular transform.";
      return;
    }
    _restPositions.resize(_source.positions.size());
    for (size_t i = 0; i < vertexCount; ++i) {
      float3 const position = TransformPoint(
          _rootFromNode,
          float3{
              _source.positions[3 * i],
              _source.positions[3 * i + 1],
              _source.positions[3 * i + 2]});
      std::memcpy(&_restPositions[3 * i], &position, sizeof(position));
    }
  }

  // Draws @p instance, a new instance of the validated asset, from a vertex buffer of its own.
  Slot PrepareSlot(filament::gltfio::FilamentInstance* instance) {
    auto& rcm = GetEngine().getRenderableManager();
    Slot slot;
    slot.instance = instance;
    [[maybe_unused]] size_t const renderableCount = FindRenderable(rcm, *instance, slot.entity);
    MOCHI_ASSERT_VERBOSE(renderableCount == 1);
    if (_indexBuffer == nullptr) {
      _indexBuffer = CreateIndexBuffer(GetEngine(), mochi::MakeConstSpan(_source.indices));
    }
    size_t const vertexCount = _source.positions.size() / 3;
    using Attribute = filament::VertexAttribute;
    using Type = filament::VertexBuffer::AttributeType;
    // A deformable GLB's primitive has no TEXCOORD_1, so a material that samples UV1 reads UV0.
    slot.vertexBuffer = filament::VertexBuffer::Builder()
                            .vertexCount(static_cast<uint32_t>(vertexCount))
                            .bufferCount(4)
                            .attribute(Attribute::POSITION, 0, Type::FLOAT3)
                            .attribute(Attribute::TANGENTS, 1, Type::SHORT4)
                            .normalized(Attribute::TANGENTS)
                            .attribute(Attribute::UV0, 2, Type::FLOAT2)
                            .attribute(Attribute::UV1, 2, Type::FLOAT2)
                            .attribute(Attribute::COLOR, 3, Type::FLOAT4)
                            .build(GetEngine());
    UploadCopy(GetEngine(), *slot.vertexBuffer, 2, _source.texcoords);
    UploadCopy(GetEngine(), *slot.vertexBuffer, 3, _source.colors);
    slot.positions.resize(vertexCount);
    slot.tangents.resize(vertexCount);
    // gltfio does not build STATIC renderables, so the primitive's buffers can be replaced.
    rcm.setGeometryAt(
        rcm.getInstance(slot.entity),
        0,
        filament::RenderableManager::PrimitiveType::TRIANGLES,
        slot.vertexBuffer,
        _indexBuffer,
        0,
        _source.indices.size());
    return slot;
  }

  // Returns the slot at @p slotIndex to the rest pose and its instance to its default state.
  void Reset(int slotIndex) {
    Slot& slot = _slots[slotIndex];
    std::memcpy(
        slot.positions.data(), _source.positions.data(), _source.positions.size() * sizeof(float));
    UpdateTangents(slot, _source.normals.data());
    _model.ResetInstanceState(slot.instance);
    Upload(slot);
  }

  // Tangent frames from @p normals, oriented by TEXCOORD_0 when the GLB has it so normal maps
  // shade correctly.
  void UpdateTangents(Slot& slot, float const* normals) const {
    size_t const vertexCount = slot.positions.size();
    filament::geometry::SurfaceOrientation::Builder builder;
    builder.vertexCount(vertexCount).normals(reinterpret_cast<float3 const*>(normals));
    if (_hasTexcoords) {
      builder.uvs(reinterpret_cast<float2 const*>(_source.texcoords.data()))
          .positions(slot.positions.data())
          .triangleCount(_source.indices.size() / 3)
          .triangles(reinterpret_cast<uint3 const*>(_source.indices.data()));
    }
    filament::geometry::SurfaceOrientation* const orientation = builder.build();
    MOCHI_ASSERT_VERBOSE(orientation != nullptr);
    orientation->getQuats(slot.tangents.data(), vertexCount);
    delete orientation;
  }

  void Upload(Slot const& slot) {
    UploadCopy(GetEngine(), *slot.vertexBuffer, 0, slot.positions);
    UploadCopy(GetEngine(), *slot.vertexBuffer, 1, slot.tangents);
    auto& rcm = GetEngine().getRenderableManager();
    rcm.setAxisAlignedBoundingBox(
        rcm.getInstance(slot.entity),
        ComputeBoundingBox(
            reinterpret_cast<float const*>(slot.positions.data()), slot.positions.size()));
  }

  RenderModel& _model;
  DeformableSourceData _source;
  // Between the mesh node's frame, which the vertex buffers hold, and the model root's, which the
  // public API uses.
  mat4f _rootFromNode;
  mat4f _nodeFromRoot;
  bool _hasTexcoords = false;
  std::vector<float> _restPositions;
  filament::IndexBuffer* _indexBuffer = nullptr;
  std::vector<Slot> _slots;
  std::vector<float3> _localNormals;
};

void RenderModel::DeformationStateDeleter::operator()(DeformationState* state) const {
  delete state;
}

std::optional<DeformableMeshView> RenderModel::GetDeformableMesh(mochi::Error& error) {
  MOCHI_ERROR_RETURN(error, std::nullopt);
  if (_ownedVertexBuffer != nullptr) {
    MOCHI_ERROR_SET(error, "UpdateGeometry has replaced the render model's geometry.");
    return std::nullopt;
  }
  if (!_glbHash.has_value()) {
    MOCHI_ERROR_SET(error, "Only render models loaded from a binary GLB file can be deformed.");
    return std::nullopt;
  }
  if (_deformation == nullptr) {
    utils::Path const file = GetPathForGltfAsset(GetPath().ToString());
    _deformation.reset(new DeformationState(
        *this,
        file.isEmpty() ? DeformableSourceData{.error = "The GLB file could not be found."}
                       : ReadDeformableGlb(file.c_str(), *_glbHash)));
  }
  if (char const* const reason = _deformation->GetError()) {
    error.SetFirstError(reason, __FILE__, __LINE__);
    return std::nullopt;
  }
  return _deformation->GetMesh();
}

std::unique_ptr<DeformableModelInstance> RenderModel::GetDeformableInstance(mochi::Error& error) {
  if (!GetDeformableMesh(error).has_value()) {
    return nullptr;
  }
  return _deformation->Checkout(error);
}

void RenderModel::ReleaseDeformableInstance(int slot) {
  _deformation->Release(slot);
}

void RenderModel::SetDeformableVertices(
    int slot,
    mochi::Span<float const> positions,
    mochi::Span<float const> normals) {
  _deformation->SetVertices(slot, positions, normals);
}

void RenderModel::CopyDeformableVertices(int from, int to) {
  _deformation->Copy(from, to);
}

int RenderModel::GetDeformableVertexCount() const {
  return _deformation->GetVertexCount();
}

int RenderModel::GetDeformableInstanceCount() const {
  return _deformation != nullptr ? _deformation->GetInstanceCount() : 0;
}

int RenderModel::GetDeformableSlotCount() const {
  return _deformation != nullptr ? _deformation->GetSlotCount() : 0;
}

//--------------------------------------------------------------------------------------------------
// DEFORMABLE MODEL INSTANCE
//--------------------------------------------------------------------------------------------------

DeformableModelInstance::DeformableModelInstance(
    RenderModel* model,
    filament::gltfio::FilamentInstance* instance,
    int slot)
    : RenderModelInstance(model, instance, -1), _slot(slot) {}

DeformableModelInstance::~DeformableModelInstance() {
  GetRenderModel()->ReleaseDeformableInstance(_slot);
}

void DeformableModelInstance::SetVertices(
    mochi::Span<float const> positions,
    mochi::Span<float const> normals) {
  GetRenderModel()->SetDeformableVertices(_slot, positions, normals);
}

int DeformableModelInstance::GetVertexCount() const {
  return GetRenderModel()->GetDeformableVertexCount();
}

std::unique_ptr<SceneObject> DeformableModelInstance::GetInstance() {
  mochi::ErrorLog error;
  std::unique_ptr<DeformableModelInstance> copy = GetRenderModel()->GetDeformableInstance(error);
  if (copy != nullptr) {
    GetRenderModel()->CopyDeformableVertices(_slot, copy->_slot);
  }
  return copy;
}

int DeformableModelInstance::GetInstanceCount() const {
  return GetRenderModel()->GetDeformableInstanceCount();
}

} // namespace mochi_renderer
