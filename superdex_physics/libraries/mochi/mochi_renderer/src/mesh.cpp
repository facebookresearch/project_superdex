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

#include <mochi_renderer/mesh.h>

#include "mesh_buffers.h"

#include <filament/IndexBuffer.h>
#include <filament/Material.h>
#include <filament/MaterialInstance.h>
#include <filament/RenderableManager.h>
#include <filament/TransformManager.h>
#include <filament/VertexBuffer.h>
#include <geometry/SurfaceOrientation.h>
#include <math/mat4.h>
#include <math/vec3.h>
#include <math/vec4.h>
#include <utils/EntityManager.h>

#include <mochi_core/geometry/model_data.h>
#include <mochi_core/geometry/tetrahedral_mesh.h>
#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/coordinate_space_converter.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/nd_array_utils.h>

#include <mochi_renderer/render_space.h>
#include <mochi_renderer/utils.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace mochi_renderer {

//--------------------------------------------------------------------------------------------------
// FILAMENT BUFFER HELPERS
//--------------------------------------------------------------------------------------------------

// Allocates a heap copy of the flat positions as float3 (caller transfers ownership to a
// Filament BufferDescriptor that frees it).
static filament::math::float3* CopyPositions(
    mochi::Span<float const> positions,
    size_t vertexCount) {
  auto* posCopy = new filament::math::float3[vertexCount];
  std::memcpy(posCopy, positions.data(), vertexCount * sizeof(filament::math::float3));
  return posCopy;
}

// Packs per-vertex normals into tangent quaternions (Filament has no NORMAL attribute),
// returning a heap array the caller transfers to a Filament BufferDescriptor.
static filament::math::short4* ComputeTangents(
    mochi::Span<float const> normals,
    size_t vertexCount) {
  auto const* normalVecs = reinterpret_cast<filament::math::float3 const*>(normals.data());
  auto* orientation = filament::geometry::SurfaceOrientation::Builder()
                          .vertexCount(vertexCount)
                          .normals(normalVecs)
                          .build();
  MOCHI_ASSERT(orientation != nullptr);
  auto* tanCopy = new filament::math::short4[vertexCount];
  orientation->getQuats(tanCopy, vertexCount);
  delete orientation;
  return tanCopy;
}

filament::IndexBuffer* CreateIndexBuffer(filament::Engine& engine, mochi::Span<int const> indices) {
  size_t const indexCount = indices.size();
  auto* indexBuffer = filament::IndexBuffer::Builder()
                          .indexCount(static_cast<uint32_t>(indexCount))
                          .bufferType(filament::IndexBuffer::IndexType::UINT)
                          .build(engine);
  auto* idxCopy = new uint32_t[indexCount];
  std::memcpy(idxCopy, indices.data(), indexCount * sizeof(uint32_t));
  indexBuffer->setBuffer(
      engine,
      filament::IndexBuffer::BufferDescriptor(
          idxCopy, indexCount * sizeof(uint32_t), [](void* buf, size_t, void*) {
            delete[] static_cast<uint32_t*>(buf);
          }));
  return indexBuffer;
}

static void SetPositionBuffer(
    filament::Engine& engine,
    filament::VertexBuffer& vertexBuffer,
    int slot,
    filament::math::float3* posCopy,
    size_t vertexCount) {
  vertexBuffer.setBufferAt(
      engine,
      slot,
      filament::VertexBuffer::BufferDescriptor(
          posCopy, vertexCount * sizeof(filament::math::float3), [](void* buf, size_t, void*) {
            delete[] static_cast<filament::math::float3*>(buf);
          }));
}

static void SetTangentBuffer(
    filament::Engine& engine,
    filament::VertexBuffer& vertexBuffer,
    int slot,
    filament::math::short4* tanCopy,
    size_t vertexCount) {
  vertexBuffer.setBufferAt(
      engine,
      slot,
      filament::VertexBuffer::BufferDescriptor(
          tanCopy, vertexCount * sizeof(filament::math::short4), [](void* buf, size_t, void*) {
            delete[] static_cast<filament::math::short4*>(buf);
          }));
}

static size_t s_meshCount = 0;

//------------------------------------------------------------------------------------------------
// UTILS
//------------------------------------------------------------------------------------------------

filament::Box ComputeBoundingBox(float const* posData, size_t vertexCount) {
  filament::math::float3 minPt{
      std::numeric_limits<float>::max(),
      std::numeric_limits<float>::max(),
      std::numeric_limits<float>::max()};
  filament::math::float3 maxPt{
      std::numeric_limits<float>::lowest(),
      std::numeric_limits<float>::lowest(),
      std::numeric_limits<float>::lowest()};

  for (size_t i = 0; i < vertexCount; ++i) {
    float const x = posData[i * 3 + 0];
    float const y = posData[i * 3 + 1];
    float const z = posData[i * 3 + 2];
    minPt.x = std::min(minPt.x, x);
    minPt.y = std::min(minPt.y, y);
    minPt.z = std::min(minPt.z, z);
    maxPt.x = std::max(maxPt.x, x);
    maxPt.y = std::max(maxPt.y, y);
    maxPt.z = std::max(maxPt.z, z);
  }

  filament::math::float3 const center = (minPt + maxPt) * 0.5f;
  filament::math::float3 const halfExtent = (maxPt - minPt) * 0.5f;
  return {center, halfExtent};
}

MeshBuffers CreateStaticMeshBuffers(
    filament::Engine& engine,
    mochi::Span<float const> positions,
    mochi::Span<float const> normals,
    mochi::Span<int const> indices) {
  size_t const vertexCount = positions.size() / 3;

  auto* vertexBuffer =
      filament::VertexBuffer::Builder()
          .vertexCount(static_cast<uint32_t>(vertexCount))
          .bufferCount(2)
          .attribute(
              filament::VertexAttribute::POSITION, 0, filament::VertexBuffer::AttributeType::FLOAT3)
          .attribute(
              filament::VertexAttribute::TANGENTS, 1, filament::VertexBuffer::AttributeType::SHORT4)
          .normalized(filament::VertexAttribute::TANGENTS)
          .build(engine);

  SetPositionBuffer(engine, *vertexBuffer, 0, CopyPositions(positions, vertexCount), vertexCount);
  SetTangentBuffer(engine, *vertexBuffer, 1, ComputeTangents(normals, vertexCount), vertexCount);

  return {
      vertexBuffer,
      CreateIndexBuffer(engine, indices),
      ComputeBoundingBox(positions.data(), vertexCount)};
}

MeshBuffers CreateModelMeshBuffers(
    filament::Engine& engine,
    mochi::Span<float const> positions,
    mochi::Span<float const> normals,
    mochi::Span<int const> indices) {
  size_t const vertexCount = positions.size() / 3;

  // gltfio's ubershader material requires COLOR/UV0/UV1 in addition to POSITION/TANGENTS,
  // even though the source mesh has none. Mirror gltfio: declare the three dummy attributes
  // sharing a single constant buffer slot (COLOR as normalized UBYTE4, UVs as normalized
  // USHORT2 — each 4 bytes/vertex, so they alias one 0xff-filled buffer = white color, (1,1)
  // UVs). Without these the renderable warns "missing required attributes".
  constexpr int kDummySlot = 2;
  auto* vertexBuffer =
      filament::VertexBuffer::Builder()
          .vertexCount(static_cast<uint32_t>(vertexCount))
          .bufferCount(3)
          .attribute(
              filament::VertexAttribute::POSITION, 0, filament::VertexBuffer::AttributeType::FLOAT3)
          .attribute(
              filament::VertexAttribute::TANGENTS, 1, filament::VertexBuffer::AttributeType::SHORT4)
          .normalized(filament::VertexAttribute::TANGENTS)
          .attribute(
              filament::VertexAttribute::COLOR,
              kDummySlot,
              filament::VertexBuffer::AttributeType::UBYTE4)
          .normalized(filament::VertexAttribute::COLOR)
          .attribute(
              filament::VertexAttribute::UV0,
              kDummySlot,
              filament::VertexBuffer::AttributeType::USHORT2)
          .normalized(filament::VertexAttribute::UV0)
          .attribute(
              filament::VertexAttribute::UV1,
              kDummySlot,
              filament::VertexBuffer::AttributeType::USHORT2)
          .normalized(filament::VertexAttribute::UV1)
          .build(engine);

  SetPositionBuffer(engine, *vertexBuffer, 0, CopyPositions(positions, vertexCount), vertexCount);
  SetTangentBuffer(engine, *vertexBuffer, 1, ComputeTangents(normals, vertexCount), vertexCount);

  // One 4-byte/vertex constant buffer (0xff) shared by COLOR, UV0 and UV1.
  size_t const dummyBytes = vertexCount * sizeof(uint32_t);
  auto* dummyData = new uint32_t[vertexCount];
  std::memset(dummyData, 0xff, dummyBytes);
  vertexBuffer->setBufferAt(
      engine,
      kDummySlot,
      filament::VertexBuffer::BufferDescriptor(dummyData, dummyBytes, [](void* buf, size_t, void*) {
        delete[] static_cast<uint32_t*>(buf);
      }));

  return {
      vertexBuffer,
      CreateIndexBuffer(engine, indices),
      ComputeBoundingBox(positions.data(), vertexCount)};
}

//--------------------------------------------------------------------------------------------------
// MESH
//--------------------------------------------------------------------------------------------------

std::unique_ptr<Mesh> Mesh::CreateMesh(
    filament::Engine* engine,
    mochi::Span<float const> positions,
    mochi::Span<float const> normals,
    mochi::Span<int const> indices,
    std::shared_ptr<MaterialInstance> material,
    bool isDynamic,
    bool isClosed) {
  MOCHI_ASSERT(!positions.empty());
  MOCHI_ASSERT(positions.size() % 3 == 0, "Expected 3 per triangle");
  MOCHI_ASSERT(normals.size() == positions.size(), "Expected one normal vector per vertex");
  auto mesh = std::unique_ptr<Mesh>(
      new Mesh(engine, positions, normals, indices, material, isDynamic, isClosed));
  mesh->SetName("Mesh" + std::to_string(s_meshCount++));
  return mesh;
}

Mesh::Mesh(
    filament::Engine* engine,
    mochi::Span<float const> positions,
    mochi::Span<float const> normals,
    mochi::Span<int const> indices,
    std::shared_ptr<MaterialInstance> material,
    bool isDynamic,
    bool isClosed)
    : SceneObject(engine) {
  size_t const vertexCount = positions.size() / 3;
  size_t const indexCount = indices.size();

  auto& em = utils::EntityManager::get();
  _engine = engine;
  _isDynamic = isDynamic;
  _isClosed = isClosed;
  _vertexCount = vertexCount;
  _indexCount = indexCount;
  _entity = em.create();
  _material = material;

  if (isDynamic) {
    // Compute tangent quaternions from normals using SurfaceOrientation.
    auto const* normalVecs = reinterpret_cast<filament::math::float3 const*>(normals.data());
    auto* orientation = filament::geometry::SurfaceOrientation::Builder()
                            .vertexCount(vertexCount)
                            .normals(normalVecs)
                            .build();
    MOCHI_ASSERT(orientation != nullptr);

    _vertexBuffer = filament::VertexBuffer::Builder()
                        .vertexCount(static_cast<uint32_t>(vertexCount))
                        .bufferCount(2)
                        .attribute(
                            filament::VertexAttribute::POSITION,
                            0,
                            filament::VertexBuffer::AttributeType::FLOAT3)
                        .attribute(
                            filament::VertexAttribute::TANGENTS,
                            1,
                            filament::VertexBuffer::AttributeType::SHORT4)
                        .normalized(filament::VertexAttribute::TANGENTS)
                        .build(*_engine);

    _dynamicPositions.resize(vertexCount);
    std::memcpy(
        _dynamicPositions.data(), positions.data(), vertexCount * sizeof(filament::math::float3));
    _dynamicTangents.resize(vertexCount);
    orientation->getQuats(_dynamicTangents.data(), vertexCount);
    delete orientation;

    _vertexBuffer->setBufferAt(
        *_engine,
        0,
        filament::VertexBuffer::BufferDescriptor(
            _dynamicPositions.data(), vertexCount * sizeof(filament::math::float3)));
    _vertexBuffer->setBufferAt(
        *_engine,
        1,
        filament::VertexBuffer::BufferDescriptor(
            _dynamicTangents.data(), vertexCount * sizeof(filament::math::short4)));

    _indexBuffer = filament::IndexBuffer::Builder()
                       .indexCount(static_cast<uint32_t>(indexCount))
                       .bufferType(filament::IndexBuffer::IndexType::UINT)
                       .build(*_engine);
    auto* idxCopy = new uint32_t[indexCount];
    std::memcpy(idxCopy, indices.data(), indexCount * sizeof(uint32_t));
    _indexBuffer->setBuffer(
        *_engine,
        filament::IndexBuffer::BufferDescriptor(
            idxCopy, indexCount * sizeof(uint32_t), [](void* buf, size_t, void*) {
              delete[] static_cast<uint32_t*>(buf);
            }));
  } else {
    MeshBuffers const buffers = CreateStaticMeshBuffers(*_engine, positions, normals, indices);
    _vertexBuffer = buffers.vertexBuffer;
    _indexBuffer = buffers.indexBuffer;
  }

  _material->Get()->setCullingMode(
      isClosed ? filament::MaterialInstance::CullingMode::BACK
               : filament::MaterialInstance::CullingMode::NONE);

  _boundingBox = ComputeBoundingBox(positions.data(), vertexCount);

  filament::RenderableManager::Builder(1)
      .boundingBox(_boundingBox)
      .material(0, _material->Get())
      .geometry(
          0,
          filament::RenderableManager::PrimitiveType::TRIANGLES,
          _vertexBuffer,
          _indexBuffer,
          0,
          indexCount)
      .culling(true)
      .receiveShadows(true)
      .castShadows(true)
      .geometryType(
          _isDynamic ? filament::RenderableManager::Builder::GeometryType::DYNAMIC
                     : filament::RenderableManager::Builder::GeometryType::STATIC)
      .build(*_engine, _entity);

  auto& tcm = engine->getTransformManager();
  tcm.setTransform(tcm.getInstance(_entity), filament::math::mat4f());
}

Mesh::~Mesh() {
  _engine->destroy(_entity);
  _material.reset();
  // Dynamic Update() enqueues buffer uploads that reference our persistent CPU buffers directly
  // (no copy), so drain the backend before freeing the buffers, or a queued update executing on the
  // engine thread could read freed memory (crash on teardown/rebuild).
  if (_isDynamic) {
    _engine->flushAndWait();
  }
  _engine->destroy(_vertexBuffer);
  _engine->destroy(_indexBuffer);
}

bool Mesh::IsDynamic() const {
  return _isDynamic;
}

bool Mesh::Update(mochi::Span<float const> positions, mochi::Span<float const> normals) {
  if (!_isDynamic) {
    return false;
  }

  MOCHI_ASSERT(positions.size() / 3 == _vertexCount);
  MOCHI_ASSERT(normals.size() == positions.size());

  auto const* normalVecs = reinterpret_cast<filament::math::float3 const*>(normals.data());
  auto* orientation = filament::geometry::SurfaceOrientation::Builder()
                          .vertexCount(_vertexCount)
                          .normals(normalVecs)
                          .build();
  MOCHI_ASSERT(orientation != nullptr);

  std::memcpy(
      _dynamicPositions.data(), positions.data(), _vertexCount * sizeof(filament::math::float3));
  orientation->getQuats(_dynamicTangents.data(), _vertexCount);
  delete orientation;

  _vertexBuffer->setBufferAt(
      *_engine,
      0,
      filament::VertexBuffer::BufferDescriptor(
          _dynamicPositions.data(), _vertexCount * sizeof(filament::math::float3)));
  _vertexBuffer->setBufferAt(
      *_engine,
      1,
      filament::VertexBuffer::BufferDescriptor(
          _dynamicTangents.data(), _vertexCount * sizeof(filament::math::short4)));

  auto& rcm = _engine->getRenderableManager();
  auto ri = rcm.getInstance(_entity);
  rcm.setAxisAlignedBoundingBox(ri, ComputeBoundingBox(positions.data(), _vertexCount));

  return true;
}

void Mesh::SetMaterial(std::shared_ptr<MaterialInstance> material) {
  auto& rcm = _engine->getRenderableManager();
  auto ri = rcm.getInstance(_entity);
  rcm.setMaterialInstanceAt(ri, 0, material->Get());
  material->Get()->setCullingMode(
      _isClosed ? filament::MaterialInstance::CullingMode::BACK
                : filament::MaterialInstance::CullingMode::NONE);
  // Assign last: the previous material instance is released only after the renderable no longer
  // references it, so it isn't destroyed while still in use.
  _material = std::move(material);
}

utils::Entity Mesh::GetRootEntity() const {
  return _entity;
}

mochi::Span<utils::Entity const> Mesh::GetEntities() const {
  return {&_entity, 1};
}

std::unique_ptr<SceneObject> Mesh::GetInstance() {
  auto& em = utils::EntityManager::get();
  auto instanceEntity = em.create();
  filament::RenderableManager::Builder(1)
      .boundingBox(_boundingBox)
      .material(0, _material->Get())
      .geometry(
          0,
          filament::RenderableManager::PrimitiveType::TRIANGLES,
          _vertexBuffer,
          _indexBuffer,
          0,
          _indexCount)
      .culling(true)
      .receiveShadows(true)
      .castShadows(true)
      .geometryType(
          _isDynamic ? filament::RenderableManager::Builder::GeometryType::DYNAMIC
                     : filament::RenderableManager::Builder::GeometryType::STATIC)
      .build(*_engine, instanceEntity);
  auto& tcm = _engine->getTransformManager();
  tcm.setTransform(tcm.getInstance(instanceEntity), filament::math::mat4f());
  _instanceCount++;
  return std::unique_ptr<MeshInstance>(new MeshInstance(_engine, instanceEntity, this));
}

int Mesh::GetInstanceCount() const {
  return _instanceCount;
}

IInstanceable* Mesh::GetInstanceable() {
  return this;
}

//--------------------------------------------------------------------------------------------------
// MESH INSTANCE
//--------------------------------------------------------------------------------------------------

MeshInstance::MeshInstance(filament::Engine* engine, utils::Entity entity, Mesh* primaryMesh)
    : SceneObject(engine), _primaryMesh(primaryMesh), _entity(entity) {}

MeshInstance::~MeshInstance() {
  _primaryMesh->_instanceCount--;
  _engine->destroy(_entity);
}

utils::Entity MeshInstance::GetRootEntity() const {
  return _entity;
}

mochi::Span<utils::Entity const> MeshInstance::GetEntities() const {
  return {&_entity, 1};
}

IInstanceable* MeshInstance::GetInstanceable() {
  return _primaryMesh;
}

//--------------------------------------------------------------------------------------------------
// LINE SEGMENT MESH
//--------------------------------------------------------------------------------------------------

namespace {
size_t s_lineSegmentMeshCount = 0;
} // namespace

std::unique_ptr<LineSegmentMesh> LineSegmentMesh::CreateLineSegmentMesh(
    filament::Engine* engine,
    mochi::Span<float const> positions,
    mochi::Span<int const> indices,
    std::shared_ptr<MaterialInstance> material,
    bool isDynamic) {
  MOCHI_ASSERT(!positions.empty());
  MOCHI_ASSERT(positions.size() % 3 == 0, "Expected 3 per vertex");
  MOCHI_ASSERT(indices.size() % 2 == 0, "Expected 2 per line segment");
  auto mesh = std::unique_ptr<LineSegmentMesh>(
      new LineSegmentMesh(engine, positions, indices, material, isDynamic));
  mesh->SetName("LineSegmentMesh" + std::to_string(s_lineSegmentMeshCount++));
  return mesh;
}

LineSegmentMesh::LineSegmentMesh(
    filament::Engine* engine,
    mochi::Span<float const> positions,
    mochi::Span<int const> indices,
    std::shared_ptr<MaterialInstance> material,
    bool isDynamic)
    : SceneObject(engine) {
  size_t const vertexCount = positions.size() / 3;
  size_t const indexCount = indices.size();

  auto& em = utils::EntityManager::get();
  _engine = engine;
  _isDynamic = isDynamic;
  _vertexCount = vertexCount;
  _indexCount = indexCount;
  _entity = em.create();
  _material = material;

  _vertexBuffer =
      filament::VertexBuffer::Builder()
          .vertexCount(static_cast<uint32_t>(vertexCount))
          .bufferCount(1)
          .attribute(
              filament::VertexAttribute::POSITION, 0, filament::VertexBuffer::AttributeType::FLOAT3)
          .build(*_engine);

  if (isDynamic) {
    _dynamicPositions.resize(vertexCount);
    std::memcpy(
        _dynamicPositions.data(), positions.data(), vertexCount * sizeof(filament::math::float3));
    _vertexBuffer->setBufferAt(
        *_engine,
        0,
        filament::VertexBuffer::BufferDescriptor(
            _dynamicPositions.data(), vertexCount * sizeof(filament::math::float3)));
  } else {
    auto* posCopy = new filament::math::float3[vertexCount];
    std::memcpy(posCopy, positions.data(), vertexCount * sizeof(filament::math::float3));
    _vertexBuffer->setBufferAt(
        *_engine,
        0,
        filament::VertexBuffer::BufferDescriptor(
            posCopy, vertexCount * sizeof(filament::math::float3), [](void* buf, size_t, void*) {
              delete[] static_cast<filament::math::float3*>(buf);
            }));
  }

  _indexBuffer = filament::IndexBuffer::Builder()
                     .indexCount(static_cast<uint32_t>(indexCount))
                     .bufferType(filament::IndexBuffer::IndexType::UINT)
                     .build(*_engine);

  auto* idxCopy = new uint32_t[indexCount];
  std::memcpy(idxCopy, indices.data(), indexCount * sizeof(uint32_t));
  _indexBuffer->setBuffer(
      *_engine,
      filament::IndexBuffer::BufferDescriptor(
          idxCopy, indexCount * sizeof(uint32_t), [](void* buf, size_t, void*) {
            delete[] static_cast<uint32_t*>(buf);
          }));

  _boundingBox = ComputeBoundingBox(positions.data(), vertexCount);

  filament::RenderableManager::Builder(1)
      .boundingBox(_boundingBox)
      .material(0, _material->Get())
      .geometry(
          0,
          filament::RenderableManager::PrimitiveType::LINES,
          _vertexBuffer,
          _indexBuffer,
          0,
          indexCount)
      .culling(false)
      .receiveShadows(false)
      .castShadows(false)
      .geometryType(
          _isDynamic ? filament::RenderableManager::Builder::GeometryType::DYNAMIC
                     : filament::RenderableManager::Builder::GeometryType::STATIC)
      .build(*_engine, _entity);

  auto& tcm = engine->getTransformManager();
  tcm.setTransform(tcm.getInstance(_entity), filament::math::mat4f());
}

LineSegmentMesh::~LineSegmentMesh() {
  _engine->destroy(_entity);
  _material.reset();
  _engine->destroy(_vertexBuffer);
  _engine->destroy(_indexBuffer);
}

bool LineSegmentMesh::IsDynamic() const {
  return _isDynamic;
}

bool LineSegmentMesh::Update(mochi::Span<float const> positions) {
  if (!_isDynamic) {
    return false;
  }
  MOCHI_ASSERT(positions.size() / 3 == _vertexCount);

  std::memcpy(
      _dynamicPositions.data(), positions.data(), _vertexCount * sizeof(filament::math::float3));
  _vertexBuffer->setBufferAt(
      *_engine,
      0,
      filament::VertexBuffer::BufferDescriptor(
          _dynamicPositions.data(), _vertexCount * sizeof(filament::math::float3)));

  auto& rcm = _engine->getRenderableManager();
  auto ri = rcm.getInstance(_entity);
  rcm.setAxisAlignedBoundingBox(ri, ComputeBoundingBox(positions.data(), _vertexCount));

  return true;
}

void LineSegmentMesh::SetMaterial(std::shared_ptr<MaterialInstance> material) {
  _material = material;
  auto& rcm = _engine->getRenderableManager();
  auto ri = rcm.getInstance(_entity);
  rcm.setMaterialInstanceAt(ri, 0, _material->Get());
}

utils::Entity LineSegmentMesh::GetRootEntity() const {
  return _entity;
}

mochi::Span<utils::Entity const> LineSegmentMesh::GetEntities() const {
  return {&_entity, 1};
}

std::unique_ptr<SceneObject> LineSegmentMesh::GetInstance() {
  if (_isDynamic) {
    MOCHI_LOG_WARNING(
        "Creating LineSegmentMeshInstance from dynamic LineSegmentMesh; updating LineSegmentMesh vertex positions will affect instances");
  }
  auto& em = utils::EntityManager::get();
  auto instanceEntity = em.create();
  filament::RenderableManager::Builder(1)
      .boundingBox(_boundingBox)
      .material(0, _material->Get())
      .geometry(
          0,
          filament::RenderableManager::PrimitiveType::LINES,
          _vertexBuffer,
          _indexBuffer,
          0,
          _indexCount)
      .culling(false)
      .receiveShadows(false)
      .castShadows(false)
      .geometryType(
          _isDynamic ? filament::RenderableManager::Builder::GeometryType::DYNAMIC
                     : filament::RenderableManager::Builder::GeometryType::STATIC)
      .build(*_engine, instanceEntity);
  auto& tcm = _engine->getTransformManager();
  tcm.setTransform(tcm.getInstance(instanceEntity), filament::math::mat4f());
  _instanceCount++;
  return std::unique_ptr<LineSegmentMeshInstance>(
      new LineSegmentMeshInstance(_engine, instanceEntity, this));
}

int LineSegmentMesh::GetInstanceCount() const {
  return _instanceCount;
}

IInstanceable* LineSegmentMesh::GetInstanceable() {
  return this;
}

//--------------------------------------------------------------------------------------------------
// LINE SEGMENT MESH INSTANCE
//--------------------------------------------------------------------------------------------------

LineSegmentMeshInstance::LineSegmentMeshInstance(
    filament::Engine* engine,
    utils::Entity entity,
    LineSegmentMesh* primaryMesh)
    : SceneObject(engine), _primaryMesh(primaryMesh), _entity(entity) {}

LineSegmentMeshInstance::~LineSegmentMeshInstance() {
  _primaryMesh->_instanceCount--;
  _engine->destroy(_entity);
}

utils::Entity LineSegmentMeshInstance::GetRootEntity() const {
  return _entity;
}

mochi::Span<utils::Entity const> LineSegmentMeshInstance::GetEntities() const {
  return {&_entity, 1};
}

IInstanceable* LineSegmentMeshInstance::GetInstanceable() {
  return _primaryMesh;
}

//--------------------------------------------------------------------------------------------------
// WIREFRAME MESH
//--------------------------------------------------------------------------------------------------

// A triangle surface rendered as a one-sided wireframe, optionally over a flat-lit surface.
// Takes the same geometry inputs as Mesh so the same data can be shared between the two. The
// caller supplies the material instances (created via ResourceManager). The triangles are
// de-indexed internally so each vertex can carry a barycentric coordinate; edges are then
// derived in the fragment shader. Back-face culling (enabled when isClosed) makes the
// wireframe visible only from the front.

namespace {
size_t s_wireframeMeshCount = 0;

// Builds the renderable for a WireframeMesh or one of its instances. The geometry,
// material instances and bounding box are shared; only the target entity differs.
// Primitive 0 is the surface pass (a depth-only pre-pass for a pure wireframe, or a flat-lit
// surface for the combined mesh); it fills the surface depth so the wireframe self-occludes.
// Primitive 1 draws the wireframe edges depth-tested against it. Shadows are enabled only for the
// combined mesh (a pure wireframe has no solid surface to cast/receive a meaningful shadow).
void BuildWireframeRenderable(
    filament::Engine& engine,
    utils::Entity entity,
    filament::Box const& boundingBox,
    filament::MaterialInstance* surfaceInstance,
    filament::MaterialInstance* wireframeInstance,
    filament::VertexBuffer* vertexBuffer,
    filament::IndexBuffer* indexBuffer,
    size_t indexCount,
    bool castShadows,
    bool isDynamic,
    size_t boneCount) {
  // GPU-skinned wireframes must not use STATIC geometry (Filament disallows skinning on STATIC),
  // and their skinned vertices can move outside the rest bounds, so treat them as DYNAMIC.
  bool const skinned = boneCount > 0;
  filament::RenderableManager::Builder builder(2);
  builder.boundingBox(boundingBox)
      .material(0, surfaceInstance)
      .geometry(
          0,
          filament::RenderableManager::PrimitiveType::TRIANGLES,
          vertexBuffer,
          indexBuffer,
          0,
          indexCount)
      .material(1, wireframeInstance)
      .geometry(
          1,
          filament::RenderableManager::PrimitiveType::TRIANGLES,
          vertexBuffer,
          indexBuffer,
          0,
          indexCount)
      .culling(true)
      .receiveShadows(castShadows)
      .castShadows(castShadows)
      .geometryType(
          (isDynamic || skinned) ? filament::RenderableManager::Builder::GeometryType::DYNAMIC
                                 : filament::RenderableManager::Builder::GeometryType::STATIC);
  if (skinned) {
    // Non-buffer skinning path: allocates `boneCount` identity bones; updated via setBones().
    builder.skinning(boneCount);
  }
  builder.build(engine, entity);

  auto& tcm = engine.getTransformManager();
  tcm.setTransform(tcm.getInstance(entity), filament::math::mat4f());
}

// Holds freshly created wireframe buffers plus their geometry counts and bounds. Ownership of the
// buffers transfers to the caller.
struct WireframeBuffers {
  filament::VertexBuffer* vertexBuffer = nullptr;
  filament::IndexBuffer* indexBuffer = nullptr;
  size_t vertexCount = 0;
  size_t indexCount = 0;
  filament::Box boundingBox;
};

// De-indexes the triangles (each corner becomes a unique vertex carrying a barycentric coordinate)
// and builds the POSITION / CUSTOM0(bary) / TANGENTS vertex buffer plus a trivial index buffer.
// When `normals` is non-empty, real tangents are derived for flat shading; otherwise identity
// tangents are used (the pure-wireframe pass ignores them). Bounds are computed from `positions`.
WireframeBuffers CreateWireframeBuffers(
    filament::Engine& engine,
    mochi::Span<float const> positions,
    mochi::Span<float const> normals,
    mochi::Span<int const> indices) {
  size_t const triangleCount = indices.size() / 3;
  size_t const vertexCount = triangleCount * 3;
  size_t const indexCount = triangleCount * 3;

  constexpr filament::math::float3 kBarycentric[3] = {
      {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}};

  auto* posData = new filament::math::float3[vertexCount];
  auto* baryData = new filament::math::float3[vertexCount];
  auto* idxData = new uint32_t[indexCount];
  // Per-corner normals matching the de-indexed positions, used to derive the tangent frame.
  std::vector<float> deindexedNormals(vertexCount * 3);
  for (size_t t = 0; t < triangleCount; ++t) {
    for (int c = 0; c < 3; ++c) {
      int const src = indices[t * 3 + c];
      size_t const dst = t * 3 + c;
      posData[dst] = {positions[src * 3], positions[src * 3 + 1], positions[src * 3 + 2]};
      baryData[dst] = kBarycentric[c];
      idxData[dst] = static_cast<uint32_t>(dst);
      deindexedNormals[dst * 3] = normals[src * 3];
      deindexedNormals[dst * 3 + 1] = normals[src * 3 + 1];
      deindexedNormals[dst * 3 + 2] = normals[src * 3 + 2];
    }
  }

  auto* vertexBuffer =
      filament::VertexBuffer::Builder()
          .vertexCount(static_cast<uint32_t>(vertexCount))
          .bufferCount(3)
          .attribute(
              filament::VertexAttribute::POSITION, 0, filament::VertexBuffer::AttributeType::FLOAT3)
          .attribute(
              filament::VertexAttribute::CUSTOM0, 1, filament::VertexBuffer::AttributeType::FLOAT3)
          .attribute(
              filament::VertexAttribute::TANGENTS, 2, filament::VertexBuffer::AttributeType::SHORT4)
          .normalized(filament::VertexAttribute::TANGENTS)
          .build(engine);
  vertexBuffer->setBufferAt(
      engine,
      0,
      filament::VertexBuffer::BufferDescriptor(
          posData, vertexCount * sizeof(filament::math::float3), [](void* buf, size_t, void*) {
            delete[] static_cast<filament::math::float3*>(buf);
          }));
  vertexBuffer->setBufferAt(
      engine,
      1,
      filament::VertexBuffer::BufferDescriptor(
          baryData, vertexCount * sizeof(filament::math::float3), [](void* buf, size_t, void*) {
            delete[] static_cast<filament::math::float3*>(buf);
          }));
  // Real tangents derived from the per-vertex normals so the flat-lit surface pass shades
  // correctly; the wireframe pass ignores them.
  filament::math::short4* tanData = ComputeTangents(
      mochi::Span<float const>(deindexedNormals.data(), deindexedNormals.size()), vertexCount);
  vertexBuffer->setBufferAt(
      engine,
      2,
      filament::VertexBuffer::BufferDescriptor(
          tanData, vertexCount * sizeof(filament::math::short4), [](void* buf, size_t, void*) {
            delete[] static_cast<filament::math::short4*>(buf);
          }));

  auto* indexBuffer = filament::IndexBuffer::Builder()
                          .indexCount(static_cast<uint32_t>(indexCount))
                          .bufferType(filament::IndexBuffer::IndexType::UINT)
                          .build(engine);
  indexBuffer->setBuffer(
      engine,
      filament::IndexBuffer::BufferDescriptor(
          idxData, indexCount * sizeof(uint32_t), [](void* buf, size_t, void*) {
            delete[] static_cast<uint32_t*>(buf);
          }));

  return {
      vertexBuffer,
      indexBuffer,
      vertexCount,
      indexCount,
      ComputeBoundingBox(positions.data(), positions.size() / 3)};
}

// De-indexes the triangles like CreateWireframeBuffers, but additionally builds BONE_INDICES
// (USHORT4, integer) and BONE_WEIGHTS (FLOAT4, normalized) attributes for GPU skinning.
// `boneIndices` / `boneWeights` are per-NODE (indexed like positions), `weightsPerVertex` entries
// each; they are de-indexed per corner and padded to 4 influences. The bounds are inflated so posed
// (skinned) vertices are not culled when they move outside the rest AABB.
WireframeBuffers CreateSkinnedWireframeBuffers(
    filament::Engine& engine,
    mochi::Span<float const> positions,
    mochi::Span<float const> normals,
    mochi::Span<int const> indices,
    mochi::Span<int const> boneIndices,
    mochi::Span<float const> boneWeights,
    int weightsPerVertex) {
  size_t const triangleCount = indices.size() / 3;
  size_t const vertexCount = triangleCount * 3;
  size_t const indexCount = triangleCount * 3;
  int const wpv = std::min(weightsPerVertex, 4);

  constexpr filament::math::float3 kBarycentric[3] = {
      {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}};

  auto* posData = new filament::math::float3[vertexCount];
  auto* baryData = new filament::math::float3[vertexCount];
  auto* boneIdxData = new filament::math::ushort4[vertexCount];
  auto* boneWtData = new filament::math::float4[vertexCount];
  auto* idxData = new uint32_t[indexCount];
  std::vector<float> deindexedNormals(vertexCount * 3);
  for (size_t t = 0; t < triangleCount; ++t) {
    for (int c = 0; c < 3; ++c) {
      int const src = indices[t * 3 + c];
      size_t const dst = t * 3 + c;
      posData[dst] = {positions[src * 3], positions[src * 3 + 1], positions[src * 3 + 2]};
      baryData[dst] = kBarycentric[c];
      idxData[dst] = static_cast<uint32_t>(dst);
      deindexedNormals[dst * 3] = normals[src * 3];
      deindexedNormals[dst * 3 + 1] = normals[src * 3 + 1];
      deindexedNormals[dst * 3 + 2] = normals[src * 3 + 2];
      filament::math::ushort4 bi{0, 0, 0, 0};
      filament::math::float4 bw{0.0f, 0.0f, 0.0f, 0.0f};
      for (int k = 0; k < wpv; ++k) {
        bi[k] = static_cast<uint16_t>(boneIndices[src * weightsPerVertex + k]);
        bw[k] = boneWeights[src * weightsPerVertex + k];
      }
      boneIdxData[dst] = bi;
      boneWtData[dst] = bw;
    }
  }

  auto* vertexBuffer =
      filament::VertexBuffer::Builder()
          .vertexCount(static_cast<uint32_t>(vertexCount))
          .bufferCount(5)
          .attribute(
              filament::VertexAttribute::POSITION, 0, filament::VertexBuffer::AttributeType::FLOAT3)
          .attribute(
              filament::VertexAttribute::CUSTOM0, 1, filament::VertexBuffer::AttributeType::FLOAT3)
          .attribute(
              filament::VertexAttribute::TANGENTS, 2, filament::VertexBuffer::AttributeType::SHORT4)
          .normalized(filament::VertexAttribute::TANGENTS)
          .attribute(
              filament::VertexAttribute::BONE_INDICES,
              3,
              filament::VertexBuffer::AttributeType::USHORT4)
          .attribute(
              filament::VertexAttribute::BONE_WEIGHTS,
              4,
              filament::VertexBuffer::AttributeType::FLOAT4)
          .build(engine);
  vertexBuffer->setBufferAt(
      engine,
      0,
      filament::VertexBuffer::BufferDescriptor(
          posData, vertexCount * sizeof(filament::math::float3), [](void* buf, size_t, void*) {
            delete[] static_cast<filament::math::float3*>(buf);
          }));
  vertexBuffer->setBufferAt(
      engine,
      1,
      filament::VertexBuffer::BufferDescriptor(
          baryData, vertexCount * sizeof(filament::math::float3), [](void* buf, size_t, void*) {
            delete[] static_cast<filament::math::float3*>(buf);
          }));
  filament::math::short4* tanData = ComputeTangents(
      mochi::Span<float const>(deindexedNormals.data(), deindexedNormals.size()), vertexCount);
  vertexBuffer->setBufferAt(
      engine,
      2,
      filament::VertexBuffer::BufferDescriptor(
          tanData, vertexCount * sizeof(filament::math::short4), [](void* buf, size_t, void*) {
            delete[] static_cast<filament::math::short4*>(buf);
          }));
  vertexBuffer->setBufferAt(
      engine,
      3,
      filament::VertexBuffer::BufferDescriptor(
          boneIdxData, vertexCount * sizeof(filament::math::ushort4), [](void* buf, size_t, void*) {
            delete[] static_cast<filament::math::ushort4*>(buf);
          }));
  vertexBuffer->setBufferAt(
      engine,
      4,
      filament::VertexBuffer::BufferDescriptor(
          boneWtData, vertexCount * sizeof(filament::math::float4), [](void* buf, size_t, void*) {
            delete[] static_cast<filament::math::float4*>(buf);
          }));

  auto* indexBuffer = filament::IndexBuffer::Builder()
                          .indexCount(static_cast<uint32_t>(indexCount))
                          .bufferType(filament::IndexBuffer::IndexType::UINT)
                          .build(engine);
  indexBuffer->setBuffer(
      engine,
      filament::IndexBuffer::BufferDescriptor(
          idxData, indexCount * sizeof(uint32_t), [](void* buf, size_t, void*) {
            delete[] static_cast<uint32_t*>(buf);
          }));

  // Tight rest AABB. SetBoneMatrices recomputes the renderable AABB per pose from these bounds and
  // the bone matrices, so skinned deformation is neither culled nor able to drag the computed scene
  // floor far below the actual geometry (a static inflated box would do both).
  filament::Box const box = ComputeBoundingBox(positions.data(), positions.size() / 3);

  return {vertexBuffer, indexBuffer, vertexCount, indexCount, box};
}
} // namespace

std::unique_ptr<WireframeMesh> WireframeMesh::CreateWireframeMesh(
    filament::Engine* engine,
    mochi::Span<float const> positions,
    mochi::Span<float const> normals,
    mochi::Span<int const> indices,
    std::shared_ptr<MaterialInstance> wireframeMaterial,
    std::shared_ptr<MaterialInstance> surfaceMaterial,
    bool isClosed,
    bool castShadows,
    bool isDynamic) {
  MOCHI_ASSERT(!positions.empty());
  MOCHI_ASSERT(positions.size() % 3 == 0, "Expected 3 floats per vertex");
  MOCHI_ASSERT(normals.size() == positions.size(), "Expected one normal per vertex");
  MOCHI_ASSERT(indices.size() % 3 == 0, "Expected 3 indices per triangle");
  auto mesh = std::unique_ptr<WireframeMesh>(new WireframeMesh(
      engine,
      positions,
      normals,
      indices,
      std::move(wireframeMaterial),
      std::move(surfaceMaterial),
      isClosed,
      castShadows,
      isDynamic));
  mesh->SetName("WireframeMesh" + std::to_string(s_wireframeMeshCount++));
  return mesh;
}

WireframeMesh::WireframeMesh(
    filament::Engine* engine,
    mochi::Span<float const> positions,
    mochi::Span<float const> normals,
    mochi::Span<int const> indices,
    std::shared_ptr<MaterialInstance> wireframeMaterial,
    std::shared_ptr<MaterialInstance> surfaceMaterial,
    bool isClosed,
    bool castShadows,
    bool isDynamic)
    : SceneObject(engine),
      _surfaceMaterial(std::move(surfaceMaterial)),
      _wireframeMaterial(std::move(wireframeMaterial)) {
  auto& em = utils::EntityManager::get();
  _engine = engine;
  _entity = em.create();
  _isDynamic = isDynamic;
  _castShadows = castShadows;

  WireframeBuffers const buffers = CreateWireframeBuffers(*_engine, positions, normals, indices);
  _vertexBuffer = buffers.vertexBuffer;
  _indexBuffer = buffers.indexBuffer;
  _vertexCount = buffers.vertexCount;
  _indexCount = buffers.indexCount;
  _boundingBox = buffers.boundingBox;

  if (_isDynamic) {
    // Capture the per-corner→node connectivity and persistent CPU buffers so Update() can rewrite
    // the position/tangent slots in place (the barycentric slot and index buffer are topology-fixed
    // and never change). `indices` lists the source node for each of the 3 corners of every
    // triangle, i.e. exactly the per-corner→node map after de-indexing.
    _connectivity.assign(indices.begin(), indices.end());
    _dynamicPositions.resize(_connectivity.size());
    _dynamicTangents.resize(_connectivity.size());
  }

  // Back-face culling removes back-facing edges. Open meshes have no meaningful front/back, so
  // they fall back to drawing both sides. A double-sided surface material ignores this.
  filament::MaterialInstance::CullingMode const cullingMode = isClosed
      ? filament::MaterialInstance::CullingMode::BACK
      : filament::MaterialInstance::CullingMode::NONE;
  _wireframeMaterial->Get()->setCullingMode(cullingMode);
  _surfaceMaterial->Get()->setCullingMode(cullingMode);

  BuildWireframeRenderable(
      *_engine,
      _entity,
      _boundingBox,
      _surfaceMaterial->Get(),
      _wireframeMaterial->Get(),
      _vertexBuffer,
      _indexBuffer,
      _indexCount,
      _castShadows,
      _isDynamic,
      /*boneCount=*/0);
}

std::unique_ptr<WireframeMesh> WireframeMesh::CreateSkinnedWireframeMesh(
    filament::Engine* engine,
    mochi::Span<float const> positions,
    mochi::Span<float const> normals,
    mochi::Span<int const> indices,
    mochi::Span<int const> boneIndices,
    mochi::Span<float const> boneWeights,
    int weightsPerVertex,
    int boneCount,
    std::shared_ptr<MaterialInstance> wireframeMaterial,
    std::shared_ptr<MaterialInstance> surfaceMaterial,
    bool isClosed) {
  MOCHI_ASSERT(!positions.empty());
  MOCHI_ASSERT(positions.size() % 3 == 0, "Expected 3 floats per vertex");
  MOCHI_ASSERT(normals.size() == positions.size(), "Expected one normal per vertex");
  MOCHI_ASSERT(indices.size() % 3 == 0, "Expected 3 indices per triangle");
  MOCHI_ASSERT(weightsPerVertex > 0, "Expected at least one weight per vertex");
  // Only 4 influences per vertex fit the GPU vertex attributes; more would be silently dropped
  // without renormalizing, leaving weights that no longer sum to 1 and a visibly under-deformed
  // skin.
  MOCHI_ASSERT_VERBOSE(
      weightsPerVertex <= 4, "Expected at most 4 weights per vertex (GPU skinning limit)");
  MOCHI_ASSERT(
      boneIndices.size() == (positions.size() / 3) * static_cast<size_t>(weightsPerVertex),
      "Expected weightsPerVertex bone indices per node");
  MOCHI_ASSERT(boneWeights.size() == boneIndices.size(), "Expected one bone weight per bone index");
  MOCHI_ASSERT(boneCount > 0 && boneCount <= 255, "Expected 1..255 bones");
  auto mesh = std::unique_ptr<WireframeMesh>(new WireframeMesh(
      engine,
      positions,
      normals,
      indices,
      boneIndices,
      boneWeights,
      weightsPerVertex,
      boneCount,
      std::move(wireframeMaterial),
      std::move(surfaceMaterial),
      isClosed));
  mesh->SetName("SkinnedWireframeMesh" + std::to_string(s_wireframeMeshCount++));
  return mesh;
}

WireframeMesh::WireframeMesh(
    filament::Engine* engine,
    mochi::Span<float const> positions,
    mochi::Span<float const> normals,
    mochi::Span<int const> indices,
    mochi::Span<int const> boneIndices,
    mochi::Span<float const> boneWeights,
    int weightsPerVertex,
    int boneCount,
    std::shared_ptr<MaterialInstance> wireframeMaterial,
    std::shared_ptr<MaterialInstance> surfaceMaterial,
    bool isClosed)
    : SceneObject(engine),
      _surfaceMaterial(std::move(surfaceMaterial)),
      _wireframeMaterial(std::move(wireframeMaterial)) {
  auto& em = utils::EntityManager::get();
  _engine = engine;
  _entity = em.create();
  _isDynamic = false;
  _castShadows = true;
  _boneCount = static_cast<size_t>(boneCount);

  WireframeBuffers const buffers = CreateSkinnedWireframeBuffers(
      *_engine, positions, normals, indices, boneIndices, boneWeights, weightsPerVertex);
  _vertexBuffer = buffers.vertexBuffer;
  _indexBuffer = buffers.indexBuffer;
  _vertexCount = buffers.vertexCount;
  _indexCount = buffers.indexCount;
  _boundingBox = buffers.boundingBox;

  // Per-bone rest sub-AABBs: the bounds of just the vertices each bone influences (weight > 0), in
  // the mesh's rest/object frame. SetBoneMatrices transforms each bone's own sub-box (rather than
  // the whole mesh box) to keep the posed AABB tight. Bones with no influenced vertices stay
  // invalid (halfExtent.x < 0) and are skipped.
  _boneRestBoxes.assign(
      static_cast<size_t>(boneCount), filament::Box{{0.0f, 0.0f, 0.0f}, {-1.0f, -1.0f, -1.0f}});
  {
    std::vector<filament::math::float3> boneMin(
        static_cast<size_t>(boneCount), filament::math::float3{std::numeric_limits<float>::max()});
    std::vector<filament::math::float3> boneMax(
        static_cast<size_t>(boneCount),
        filament::math::float3{std::numeric_limits<float>::lowest()});
    size_t const vertexCount = positions.size() / 3;
    for (size_t v = 0; v < vertexCount; ++v) {
      filament::math::float3 const p{
          positions[v * 3 + 0], positions[v * 3 + 1], positions[v * 3 + 2]};
      // Only the first 4 influences reach the GPU (see CreateSkinnedWireframeBuffers), so bones
      // referenced solely in later slots must not widen a sub-box they never deform.
      for (int k = 0; k < std::min(weightsPerVertex, 4); ++k) {
        size_t const idx = v * static_cast<size_t>(weightsPerVertex) + static_cast<size_t>(k);
        float const w = boneWeights[idx];
        int const bone = boneIndices[idx];
        if (w <= 0.0f || bone < 0 || bone >= boneCount) {
          continue;
        }
        boneMin[bone] = min(boneMin[bone], p);
        boneMax[bone] = max(boneMax[bone], p);
      }
    }
    for (int b = 0; b < boneCount; ++b) {
      if (boneMin[b].x <= boneMax[b].x) {
        _boneRestBoxes[b] =
            filament::Box{(boneMin[b] + boneMax[b]) * 0.5f, (boneMax[b] - boneMin[b]) * 0.5f};
      }
    }
  }

  filament::MaterialInstance::CullingMode const cullingMode = isClosed
      ? filament::MaterialInstance::CullingMode::BACK
      : filament::MaterialInstance::CullingMode::NONE;
  _wireframeMaterial->Get()->setCullingMode(cullingMode);
  _surfaceMaterial->Get()->setCullingMode(cullingMode);

  BuildWireframeRenderable(
      *_engine,
      _entity,
      _boundingBox,
      _surfaceMaterial->Get(),
      _wireframeMaterial->Get(),
      _vertexBuffer,
      _indexBuffer,
      _indexCount,
      _castShadows,
      _isDynamic,
      _boneCount);
}

bool WireframeMesh::UpdateGeometry(
    mochi::Span<float const> positions,
    mochi::Span<float const> normals,
    mochi::Span<int const> indices) {
  if (!_isDynamic) {
    MOCHI_LOG_WARNING("WireframeMesh::UpdateGeometry called on a non-dynamic mesh; ignored.");
    return false;
  }
  if (positions.empty() || indices.empty()) {
    MOCHI_LOG_ERROR("WireframeMesh::UpdateGeometry called with empty geometry.");
    return false;
  }

  WireframeBuffers const buffers = CreateWireframeBuffers(*_engine, positions, normals, indices);

  // Re-point the primary and every live instance at the new buffers, so all instances (e.g. across
  // editors) update at once. setGeometryAt supports a changed index count.
  auto& rcm = _engine->getRenderableManager();
  auto repoint = [&](utils::Entity entity) {
    auto ri = rcm.getInstance(entity);
    if (!ri.isValid()) {
      return;
    }
    size_t const primitiveCount = rcm.getPrimitiveCount(ri);
    for (size_t p = 0; p < primitiveCount; ++p) {
      rcm.setGeometryAt(
          ri,
          p,
          filament::RenderableManager::PrimitiveType::TRIANGLES,
          buffers.vertexBuffer,
          buffers.indexBuffer,
          0,
          buffers.indexCount);
    }
    rcm.setAxisAlignedBoundingBox(ri, buffers.boundingBox);
  };
  repoint(_entity);
  for (utils::Entity const entity : _instanceEntities) {
    repoint(entity);
  }

  // Drain the backend before freeing the buffers the renderables referenced until now.
  _engine->flushAndWait();
  _engine->destroy(_vertexBuffer);
  _engine->destroy(_indexBuffer);

  _vertexBuffer = buffers.vertexBuffer;
  _indexBuffer = buffers.indexBuffer;
  _vertexCount = buffers.vertexCount;
  _indexCount = buffers.indexCount;
  _boundingBox = buffers.boundingBox;
  return true;
}

WireframeMesh::~WireframeMesh() {
  _engine->destroy(_entity);
  // Dynamic Update() enqueues buffer uploads that reference our persistent CPU buffers directly
  // (no copy), so drain the backend before freeing the buffers, or a queued update executing on the
  // engine thread could read freed memory (crash on teardown/rebuild).
  if (_isDynamic) {
    _engine->flushAndWait();
  }
  _engine->destroy(_vertexBuffer);
  _engine->destroy(_indexBuffer);
  // Material instances are owned by the caller (shared_ptr<MaterialInstance>), so they are
  // not destroyed here.
}

bool WireframeMesh::Update(mochi::Span<float const> positions, mochi::Span<float const> normals) {
  if (!_isDynamic) {
    return false;
  }

  size_t const cornerCount = _connectivity.size();
  MOCHI_ASSERT(cornerCount == _vertexCount);
  MOCHI_ASSERT(normals.size() == positions.size());

  // De-index per-node positions/normals into the per-corner vertices (corner c → node).
  if (_dynamicDeindexedNormals.size() != cornerCount * 3) {
    _dynamicDeindexedNormals.resize(cornerCount * 3);
  }
  for (size_t corner = 0; corner < cornerCount; ++corner) {
    int const node = _connectivity[corner];
    _dynamicPositions[corner] = {
        positions[node * 3], positions[node * 3 + 1], positions[node * 3 + 2]};
    _dynamicDeindexedNormals[corner * 3] = normals[node * 3];
    _dynamicDeindexedNormals[corner * 3 + 1] = normals[node * 3 + 1];
    _dynamicDeindexedNormals[corner * 3 + 2] = normals[node * 3 + 2];
  }

  auto const* normalVecs =
      reinterpret_cast<filament::math::float3 const*>(_dynamicDeindexedNormals.data());
  auto* orientation = filament::geometry::SurfaceOrientation::Builder()
                          .vertexCount(cornerCount)
                          .normals(normalVecs)
                          .build();
  MOCHI_ASSERT(orientation != nullptr);
  orientation->getQuats(_dynamicTangents.data(), cornerCount);
  delete orientation;

  _vertexBuffer->setBufferAt(
      *_engine,
      0,
      filament::VertexBuffer::BufferDescriptor(
          _dynamicPositions.data(), cornerCount * sizeof(filament::math::float3)));
  _vertexBuffer->setBufferAt(
      *_engine,
      2,
      filament::VertexBuffer::BufferDescriptor(
          _dynamicTangents.data(), cornerCount * sizeof(filament::math::short4)));

  _boundingBox =
      ComputeBoundingBox(reinterpret_cast<float const*>(_dynamicPositions.data()), cornerCount);

  // Instances share the same buffers (dynamic geometry), so they deform for free; only their
  // bounds need refreshing along with the primary.
  auto& rcm = _engine->getRenderableManager();
  auto refreshBounds = [&](utils::Entity entity) {
    auto ri = rcm.getInstance(entity);
    if (ri.isValid()) {
      rcm.setAxisAlignedBoundingBox(ri, _boundingBox);
    }
  };
  refreshBounds(_entity);
  for (utils::Entity const entity : _instanceEntities) {
    refreshBounds(entity);
  }

  return true;
}

void WireframeMesh::SetColor(filament::math::float4 color) {
  _wireframeMaterial->Get()->setParameter("color", color);
}

void WireframeMesh::SetBoneMatrices(mochi::Span<filament::math::mat4f const> bones) {
  if (_boneCount == 0 || bones.empty()) {
    return;
  }
  auto& rcm = _engine->getRenderableManager();
  size_t const count = std::min(bones.size(), _boneCount);

  // Recompute a tight posed AABB enclosing the deformed surface. Every skinned vertex is a convex
  // combination of {bone_i * v}, so unioning each bone's transform of the rest-AABB corners bounds
  // the deformation. Crucially we transform each bone's OWN sub-box (the bounds of just the
  // vertices it influences) rather than the whole mesh box: a whole-box-per-bone union over-bounds
  // badly and drags the computed scene floor far below the mesh (the floor takes the lowest AABB
  // point). Updating the AABB per pose also keeps the skin from being frustum-culled.
  filament::math::float3 minPt{
      std::numeric_limits<float>::max(),
      std::numeric_limits<float>::max(),
      std::numeric_limits<float>::max()};
  filament::math::float3 maxPt{
      std::numeric_limits<float>::lowest(),
      std::numeric_limits<float>::lowest(),
      std::numeric_limits<float>::lowest()};
  bool anyValid = false;
  auto accumulate = [&](filament::math::mat4f const& bone, filament::Box const& box) {
    filament::math::float3 const c = box.center;
    filament::math::float3 const h = box.halfExtent;
    for (int corner = 0; corner < 8; ++corner) {
      filament::math::float4 const p{
          c.x + ((corner & 1) ? h.x : -h.x),
          c.y + ((corner & 2) ? h.y : -h.y),
          c.z + ((corner & 4) ? h.z : -h.z),
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
  for (size_t b = 0; b < count && b < _boneRestBoxes.size(); ++b) {
    // Skip bones that influence no vertices (invalid sub-box, halfExtent.x < 0).
    if (_boneRestBoxes[b].halfExtent.x >= 0.0f) {
      accumulate(bones[b], _boneRestBoxes[b]);
    }
  }
  // Fallback for meshes without per-bone sub-boxes: bound each bone by the whole rest box.
  if (!anyValid) {
    for (size_t b = 0; b < count; ++b) {
      accumulate(bones[b], _boundingBox);
    }
  }
  filament::Box const posedBox{(minPt + maxPt) * 0.5f, (maxPt - minPt) * 0.5f};

  auto apply = [&](utils::Entity entity) {
    auto ri = rcm.getInstance(entity);
    if (ri.isValid()) {
      rcm.setBones(ri, bones.data(), count, 0);
      rcm.setAxisAlignedBoundingBox(ri, posedBox);
    }
  };
  apply(_entity);
  for (utils::Entity const entity : _instanceEntities) {
    apply(entity);
  }
}

void WireframeMesh::SetMaterial(std::shared_ptr<MaterialInstance> /*material*/) {
  MOCHI_LOG_WARNING("WireframeMesh manages its own material; SetMaterial is ignored.");
}

utils::Entity WireframeMesh::GetRootEntity() const {
  return _entity;
}

mochi::Span<utils::Entity const> WireframeMesh::GetEntities() const {
  return {&_entity, 1};
}

std::unique_ptr<SceneObject> WireframeMesh::GetInstance() {
  auto& em = utils::EntityManager::get();
  auto instanceEntity = em.create();
  BuildWireframeRenderable(
      *_engine,
      instanceEntity,
      _boundingBox,
      _surfaceMaterial->Get(),
      _wireframeMaterial->Get(),
      _vertexBuffer,
      _indexBuffer,
      _indexCount,
      _castShadows,
      _isDynamic,
      _boneCount);
  _instanceEntities.push_back(instanceEntity);
  return std::unique_ptr<WireframeMeshInstance>(
      new WireframeMeshInstance(_engine, instanceEntity, this));
}

int WireframeMesh::GetInstanceCount() const {
  return static_cast<int>(_instanceEntities.size());
}

IInstanceable* WireframeMesh::GetInstanceable() {
  return this;
}

//--------------------------------------------------------------------------------------------------
// WIREFRAME MESH INSTANCE
//--------------------------------------------------------------------------------------------------

WireframeMeshInstance::WireframeMeshInstance(
    filament::Engine* engine,
    utils::Entity entity,
    WireframeMesh* primaryMesh)
    : SceneObject(engine), _primaryMesh(primaryMesh), _entity(entity) {}

WireframeMeshInstance::~WireframeMeshInstance() {
  auto& entities = _primaryMesh->_instanceEntities;
  entities.erase(std::remove(entities.begin(), entities.end(), _entity), entities.end());
  _engine->destroy(_entity);
}

utils::Entity WireframeMeshInstance::GetRootEntity() const {
  return _entity;
}

mochi::Span<utils::Entity const> WireframeMeshInstance::GetEntities() const {
  return {&_entity, 1};
}

IInstanceable* WireframeMeshInstance::GetInstanceable() {
  return _primaryMesh;
}

} // namespace mochi_renderer
