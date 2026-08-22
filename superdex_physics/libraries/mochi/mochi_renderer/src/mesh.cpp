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
// PROCEDURAL MESH GENERATORS (for Mochi models defined by an analytic shape)
//--------------------------------------------------------------------------------------------------

struct ProceduralMesh {
  std::vector<float> positions;
  std::vector<float> normals;
  std::vector<int> indices;
};

static ProceduralMesh GenerateBoxMesh(mochi::Box const& box) {
  ProceduralMesh mesh;
  mesh.positions.reserve(72);
  mesh.normals.reserve(72);
  mesh.indices.reserve(36);
  auto const& c = box.center;
  auto const& h = box.halfExtents;
  auto const& q = box.rotation;
  mochi::Real3 const localCorners[8] = {
      {-h[0], -h[1], -h[2]},
      {+h[0], -h[1], -h[2]},
      {+h[0], +h[1], -h[2]},
      {-h[0], +h[1], -h[2]},
      {-h[0], -h[1], +h[2]},
      {+h[0], -h[1], +h[2]},
      {+h[0], +h[1], +h[2]},
      {-h[0], +h[1], +h[2]},
  };
  struct Face {
    int v[4];
    mochi::Real3 normal;
  };
  using R = mochi::real;
  Face const faces[6] = {
      {{1, 2, 6, 5}, {R(1), R(0), R(0)}},
      {{0, 4, 7, 3}, {R(-1), R(0), R(0)}},
      {{3, 7, 6, 2}, {R(0), R(1), R(0)}},
      {{0, 1, 5, 4}, {R(0), R(-1), R(0)}},
      {{4, 5, 6, 7}, {R(0), R(0), R(1)}},
      {{0, 3, 2, 1}, {R(0), R(0), R(-1)}},
  };
  for (auto const& face : faces) {
    int const base = static_cast<int>(mesh.positions.size()) / 3;
    mochi::Real3 const n = q * face.normal;
    for (int i = 0; i < 4; ++i) {
      mochi::Real3 const v = c + q * localCorners[face.v[i]];
      mesh.positions.push_back(static_cast<float>(v[0]));
      mesh.positions.push_back(static_cast<float>(v[1]));
      mesh.positions.push_back(static_cast<float>(v[2]));
      mesh.normals.push_back(static_cast<float>(n[0]));
      mesh.normals.push_back(static_cast<float>(n[1]));
      mesh.normals.push_back(static_cast<float>(n[2]));
    }
    mesh.indices.push_back(base);
    mesh.indices.push_back(base + 1);
    mesh.indices.push_back(base + 2);
    mesh.indices.push_back(base);
    mesh.indices.push_back(base + 2);
    mesh.indices.push_back(base + 3);
  }
  return mesh;
}

static ProceduralMesh GeneratePlaneMesh(mochi::Plane const& plane) {
  ProceduralMesh mesh;
  mesh.positions.reserve(12);
  mesh.normals.reserve(12);
  mochi::Real3 const n = mochi::Normalize(plane.normal);
  mochi::Real3 const origin = n * plane.distance;
  mochi::Real3 tangent;
  if (std::abs(n[0]) < mochi::real(0.9)) {
    tangent = mochi::Normalize(
        mochi::Cross(n, mochi::Real3{mochi::real(1), mochi::real(0), mochi::real(0)}));
  } else {
    tangent = mochi::Normalize(
        mochi::Cross(n, mochi::Real3{mochi::real(0), mochi::real(1), mochi::real(0)}));
  }
  mochi::Real3 const bitangent = mochi::Cross(n, tangent);
  mochi::real const halfSize = mochi::real(5);
  mochi::Real3 const corners[4] = {
      origin - tangent * halfSize - bitangent * halfSize,
      origin + tangent * halfSize - bitangent * halfSize,
      origin + tangent * halfSize + bitangent * halfSize,
      origin - tangent * halfSize + bitangent * halfSize,
  };
  for (auto const& v : corners) {
    mesh.positions.push_back(static_cast<float>(v[0]));
    mesh.positions.push_back(static_cast<float>(v[1]));
    mesh.positions.push_back(static_cast<float>(v[2]));
    mesh.normals.push_back(static_cast<float>(n[0]));
    mesh.normals.push_back(static_cast<float>(n[1]));
    mesh.normals.push_back(static_cast<float>(n[2]));
  }
  mesh.indices = {0, 1, 2, 0, 2, 3};
  return mesh;
}

static ProceduralMesh GenerateSphereMesh(mochi::Sphere const& sphere) {
  ProceduralMesh mesh;
  int const numLon = 32;
  int const numLat = 16;
  int const numVerts = (numLat + 1) * (numLon + 1);
  mesh.positions.reserve(numVerts * 3);
  mesh.normals.reserve(numVerts * 3);
  mesh.indices.reserve(numLat * numLon * 6);
  auto const r = static_cast<float>(sphere.radius);
  auto const cx = static_cast<float>(sphere.center[0]);
  auto const cy = static_cast<float>(sphere.center[1]);
  auto const cz = static_cast<float>(sphere.center[2]);
  for (int lat = 0; lat <= numLat; ++lat) {
    float const theta =
        static_cast<float>(mochi::kPI) * static_cast<float>(lat) / static_cast<float>(numLat);
    float const sinT = std::sin(theta);
    float const cosT = std::cos(theta);
    for (int lon = 0; lon <= numLon; ++lon) {
      float const phi = 2.0f * static_cast<float>(mochi::kPI) * static_cast<float>(lon) /
          static_cast<float>(numLon);
      float const nx = sinT * std::cos(phi);
      float const ny = sinT * std::sin(phi);
      float const nz = cosT;
      mesh.positions.push_back(cx + r * nx);
      mesh.positions.push_back(cy + r * ny);
      mesh.positions.push_back(cz + r * nz);
      mesh.normals.push_back(nx);
      mesh.normals.push_back(ny);
      mesh.normals.push_back(nz);
    }
  }
  for (int lat = 0; lat < numLat; ++lat) {
    for (int lon = 0; lon < numLon; ++lon) {
      int const curr = lat * (numLon + 1) + lon;
      int const next = curr + numLon + 1;
      if (lat != 0) {
        mesh.indices.push_back(curr);
        mesh.indices.push_back(next);
        mesh.indices.push_back(curr + 1);
      }
      if (lat != numLat - 1) {
        mesh.indices.push_back(curr + 1);
        mesh.indices.push_back(next);
        mesh.indices.push_back(next + 1);
      }
    }
  }
  return mesh;
}

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

// Builds a UINT IndexBuffer from triangle indices.
static filament::IndexBuffer* CreateIndexBuffer(
    filament::Engine& engine,
    mochi::Span<int const> indices) {
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

static filament::Box ComputeBoundingBox(float const* posData, size_t vertexCount) {
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

bool BuildMochiModelGeometry(
    mochi::ModelData const& modelData,
    mochi::CoordinateSpaceConverter const* converter,
    std::vector<float>& positions,
    std::vector<float>& vertexNormals,
    std::vector<int>& indices) {
  mochi::CoordinateSpaceConverter const defaultConverter(
      mochi::CoordinateSpace::Default(), RenderSpace());
  mochi::CoordinateSpaceConverter const& spaceConverter = converter ? *converter : defaultConverter;

  positions.clear();
  vertexNormals.clear();
  indices.clear();

  if (modelData.mesh) {
    mochi::Span<mochi::Real3 const> surfaceNodes;
    mochi::Span<mochi::Int3 const> surfaceTris;
    std::shared_ptr<mochi::TriangularMesh const> boundaryMesh;
    int const nodesPerElement = modelData.mesh->nodesPerElement;
    if (nodesPerElement == 4) {
      // Tetrahedral mesh: extract boundary surface
      auto nodes =
          mochi::Unflatten<mochi::Real3 const>(mochi::MakeConstSpan(modelData.mesh->coordinates));
      auto tets =
          mochi::Unflatten<mochi::Int4 const>(mochi::MakeConstSpan(modelData.mesh->connectivity));
      mochi::TetrahedralMesh tetMesh(nodes, tets);
      boundaryMesh = tetMesh.GetBoundaryMesh();
      surfaceNodes = boundaryMesh->GetNodeCoordinates();
      surfaceTris = boundaryMesh->GetElementConnectivity();
    } else if (nodesPerElement == 3) {
      // Surface mesh: use directly
      surfaceNodes =
          mochi::Unflatten<mochi::Real3 const>(mochi::MakeConstSpan(modelData.mesh->coordinates));
      surfaceTris =
          mochi::Unflatten<mochi::Int3 const>(mochi::MakeConstSpan(modelData.mesh->connectivity));
    } else {
      return false;
    }
    positions.reserve(surfaceNodes.size() * 3);
    for (auto const& node : surfaceNodes) {
      auto const pos = spaceConverter.TranslationToOutput(StaticCast<mochi::Float3>(node));
      positions.push_back(pos[0]);
      positions.push_back(pos[1]);
      positions.push_back(pos[2]);
    }

    indices.reserve(surfaceTris.size() * 3);
    for (auto const& tri : surfaceTris) {
      indices.push_back(tri[0]);
      indices.push_back(tri[1]);
      indices.push_back(tri[2]);
    }

    std::vector<float> faceNormals;
    faceNormals.reserve(surfaceTris.size() * 3);
    for (auto const& tri : surfaceTris) {
      auto const v0 = StaticCast<mochi::Float3>(surfaceNodes[tri[0]]);
      auto const v1 = StaticCast<mochi::Float3>(surfaceNodes[tri[1]]);
      auto const v2 = StaticCast<mochi::Float3>(surfaceNodes[tri[2]]);
      auto const e1 = v1 - v0;
      auto const e2 = v2 - v0;
      auto n = mochi::Cross(e1, e2);
      float const len = mochi::Norm(n);
      if (len > 0) {
        n = n / len;
      }
      auto const norm = spaceConverter.DirectionToOutput(n);
      faceNormals.push_back(norm[0]);
      faceNormals.push_back(norm[1]);
      faceNormals.push_back(norm[2]);
    }
    ComputeVertexNormalsAngleWeighted(positions, faceNormals, indices, vertexNormals);
  } else {
    ProceduralMesh procMesh;
    if (modelData.box) {
      procMesh = GenerateBoxMesh(*modelData.box);
    } else if (modelData.plane) {
      procMesh = GeneratePlaneMesh(*modelData.plane);
    } else if (modelData.sphere) {
      procMesh = GenerateSphereMesh(*modelData.sphere);
    } else {
      return false;
    }
    spaceConverter.TranslationsToOutput(mochi::MakeSpan(procMesh.positions), mochi::ErrorAssert{});
    spaceConverter.DirectionsToOutput(mochi::MakeSpan(procMesh.normals), mochi::ErrorAssert{});
    positions = std::move(procMesh.positions);
    vertexNormals = std::move(procMesh.normals);
    indices = std::move(procMesh.indices);
  }

  if (positions.empty() || indices.empty() || positions.size() % 3 != 0 ||
      vertexNormals.size() != positions.size()) {
    return false;
  }
  return true;
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
static size_t s_lineSegmentMeshCount = 0;
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
static size_t s_wireframeMeshCount = 0;

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
    bool isDynamic) {
  filament::RenderableManager::Builder(2)
      .boundingBox(boundingBox)
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
          isDynamic ? filament::RenderableManager::Builder::GeometryType::DYNAMIC
                    : filament::RenderableManager::Builder::GeometryType::STATIC)
      .build(engine, entity);

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
      _isDynamic);
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
      _isDynamic);
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
