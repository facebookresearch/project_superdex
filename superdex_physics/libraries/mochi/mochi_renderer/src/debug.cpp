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

#include <mochi_renderer/debug.h>

#include <filament/Box.h>
#include <filament/IndexBuffer.h>
#include <filament/MaterialInstance.h>
#include <filament/RenderableManager.h>
#include <filament/Scene.h>
#include <filament/Texture.h>
#include <filament/TextureSampler.h>
#include <filament/VertexBuffer.h>
#include <math/mat3.h>
#include <math/norm.h>
#include <utils/EntityManager.h>

#include <mochi_core/utils/debug.h>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <numeric>

#include <utils/unwindows.h>

#include "filament/Material.h"
#include "materials.h"

namespace mochi_renderer {

static constexpr int kInstancedSphereStacks = 12;
static constexpr int kInstancedSphereSlices = 24;
static constexpr uint32_t kMaxSphereInstancesPerBatch = 32767;
static constexpr uint32_t kSphereDataTexelsPerInstance = 2;
static constexpr uint32_t kSphereDataTextureWidth = 512;

static filament::math::short4 PackNormalTangentFrame(filament::math::float3 normal) {
  normal = normalize(normal);
  filament::math::float3 const tangent = normalize(
      std::abs(normal.y) < 0.999f ? cross(filament::math::float3{0.0f, 1.0f, 0.0f}, normal)
                                  : filament::math::float3{1.0f, 0.0f, 0.0f});
  filament::math::float3 const bitangent = cross(normal, tangent);
  return filament::math::packSnorm16(
      filament::math::mat3f::packTangentFrame(filament::math::mat3f{tangent, bitangent, normal})
          .xyzw);
}

static filament::math::short4* PackNormalTangentFrames(
    std::vector<filament::math::float3> const& normals) {
  auto* tangents = new filament::math::short4[normals.size()];
  std::transform(normals.begin(), normals.end(), tangents, PackNormalTangentFrame);
  return tangents;
}

DebugDraw::DebugDraw(filament::Engine* engine, filament::Scene* scene)
    : _engine(engine), _scene(scene) {
  // Material for wireframe (lines)
  _material = filament::Material::Builder()
                  .package(
                      MOCHI_RENDERER_MATERIALS_UNLITVERTEXCOLOR_DATA,
                      MOCHI_RENDERER_MATERIALS_UNLITVERTEXCOLOR_SIZE)
                  .build(*_engine);

  _sphereMaterial = filament::Material::Builder()
                        .package(
                            MOCHI_RENDERER_MATERIALS_INSTANCEDSPHERE_DATA,
                            MOCHI_RENDERER_MATERIALS_INSTANCEDSPHERE_SIZE)
                        .build(*_engine);

  // Material for solid geometry
  _solidMaterial = filament::Material::Builder()
                       .package(
                           MOCHI_RENDERER_MATERIALS_LITVERTEXCOLOR_DATA,
                           MOCHI_RENDERER_MATERIALS_LITVERTEXCOLOR_SIZE)
                       .build(*_engine);

  // Material instance for overlay geometry (no depth testing)
  // Uses the same material as solid geometry but with depth write/culling disabled
  _overlayMaterialInstance = _solidMaterial->createInstance();
  _overlayMaterialInstance->setDepthWrite(false);
  _overlayMaterialInstance->setDepthCulling(false);

  _positions.reserve(4096);
  _colors.reserve(4096);
  _indices.reserve(8192);
  _sphereData.reserve(4096 * kSphereDataTexelsPerInstance);

  _solidPositions.reserve(4096);
  _solidNormals.reserve(4096);
  _solidColors.reserve(4096);
  _solidIndices.reserve(8192);

  _overlayPositions.reserve(4096);
  _overlayNormals.reserve(4096);
  _overlayColors.reserve(4096);
  _overlayIndices.reserve(8192);

  _bufferMaxVertices = kInitialMaxVertices;
  _bufferMaxIndices = kInitialMaxIndices;
  _solidBufferMaxVertices = kInitialMaxVertices;
  _solidBufferMaxIndices = kInitialMaxIndices;
  _overlayBufferMaxVertices = kInitialMaxVertices;
  _overlayBufferMaxIndices = kInitialMaxIndices;

  // Wireframe vertex/index buffers
  _vertexBuffer =
      filament::VertexBuffer::Builder()
          .vertexCount(_bufferMaxVertices)
          .bufferCount(2)
          .attribute(
              filament::VertexAttribute::POSITION, 0, filament::VertexBuffer::AttributeType::FLOAT3)
          .attribute(
              filament::VertexAttribute::COLOR, 1, filament::VertexBuffer::AttributeType::FLOAT4)
          .build(*_engine);

  _indexBuffer = filament::IndexBuffer::Builder()
                     .indexCount(_bufferMaxIndices)
                     .bufferType(filament::IndexBuffer::IndexType::UINT)
                     .build(*_engine);

  // Solid geometry vertex/index buffers (positions, normals, colors)
  _solidVertexBuffer =
      filament::VertexBuffer::Builder()
          .vertexCount(_solidBufferMaxVertices)
          .bufferCount(3)
          .attribute(
              filament::VertexAttribute::POSITION, 0, filament::VertexBuffer::AttributeType::FLOAT3)
          .attribute(
              filament::VertexAttribute::TANGENTS, 1, filament::VertexBuffer::AttributeType::SHORT4)
          .normalized(filament::VertexAttribute::TANGENTS)
          .attribute(
              filament::VertexAttribute::COLOR, 2, filament::VertexBuffer::AttributeType::FLOAT4)
          .build(*_engine);

  _solidIndexBuffer = filament::IndexBuffer::Builder()
                          .indexCount(_solidBufferMaxIndices)
                          .bufferType(filament::IndexBuffer::IndexType::UINT)
                          .build(*_engine);

  // Overlay geometry vertex/index buffers
  _overlayVertexBuffer =
      filament::VertexBuffer::Builder()
          .vertexCount(_overlayBufferMaxVertices)
          .bufferCount(3)
          .attribute(
              filament::VertexAttribute::POSITION, 0, filament::VertexBuffer::AttributeType::FLOAT3)
          .attribute(
              filament::VertexAttribute::TANGENTS, 1, filament::VertexBuffer::AttributeType::SHORT4)
          .normalized(filament::VertexAttribute::TANGENTS)
          .attribute(
              filament::VertexAttribute::COLOR, 2, filament::VertexBuffer::AttributeType::FLOAT4)
          .build(*_engine);

  _overlayIndexBuffer = filament::IndexBuffer::Builder()
                            .indexCount(_overlayBufferMaxIndices)
                            .bufferType(filament::IndexBuffer::IndexType::UINT)
                            .build(*_engine);

  // Create wireframe entity
  _entity = utils::EntityManager::get().create();
  filament::RenderableManager::Builder(1)
      .boundingBox({{-1e6f, -1e6f, -1e6f}, {1e6f, 1e6f, 1e6f}})
      .material(0, _material->getDefaultInstance())
      .geometry(
          0, filament::RenderableManager::PrimitiveType::LINES, _vertexBuffer, _indexBuffer, 0, 0)
      .culling(false)
      .receiveShadows(false)
      .castShadows(false)
      .geometryType(filament::RenderableManager::Builder::GeometryType::DYNAMIC)
      .build(*_engine, _entity);

  // One immutable unit sphere is shared by every debug sphere instance.
  _sphereUnitPositions.reserve((kInstancedSphereStacks + 1) * (kInstancedSphereSlices + 1));
  _sphereUnitTangents.reserve((kInstancedSphereStacks + 1) * (kInstancedSphereSlices + 1));
  _sphereUnitIndices.reserve(kInstancedSphereStacks * kInstancedSphereSlices * 6);
  constexpr float k2Pi = 2.0f * std::numbers::pi_v<float>;
  constexpr float kPi = std::numbers::pi_v<float>;
  for (int stack = 0; stack <= kInstancedSphereStacks; ++stack) {
    float const phi = kPi * static_cast<float>(stack) / static_cast<float>(kInstancedSphereStacks);
    float const sinPhi = std::sin(phi);
    float const cosPhi = std::cos(phi);
    for (int slice = 0; slice <= kInstancedSphereSlices; ++slice) {
      float const theta =
          k2Pi * static_cast<float>(slice) / static_cast<float>(kInstancedSphereSlices);
      filament::math::float3 const normal{
          sinPhi * std::cos(theta), cosPhi, sinPhi * std::sin(theta)};
      _sphereUnitPositions.push_back(normal);
      _sphereUnitTangents.push_back(PackNormalTangentFrame(normal));
    }
  }
  for (int stack = 0; stack < kInstancedSphereStacks; ++stack) {
    for (int slice = 0; slice < kInstancedSphereSlices; ++slice) {
      auto const current = static_cast<uint16_t>(stack * (kInstancedSphereSlices + 1) + slice);
      auto const next = static_cast<uint16_t>(current + kInstancedSphereSlices + 1);
      auto const currentRight = static_cast<uint16_t>(current + 1);
      auto const nextRight = static_cast<uint16_t>(next + 1);
      _sphereUnitIndices.insert(
          _sphereUnitIndices.end(), {current, currentRight, next, currentRight, nextRight, next});
    }
  }

  _sphereVertexBuffer =
      filament::VertexBuffer::Builder()
          .vertexCount(static_cast<uint32_t>(_sphereUnitPositions.size()))
          .bufferCount(2)
          .attribute(
              filament::VertexAttribute::POSITION, 0, filament::VertexBuffer::AttributeType::FLOAT3)
          .attribute(
              filament::VertexAttribute::TANGENTS, 1, filament::VertexBuffer::AttributeType::SHORT4)
          .normalized(filament::VertexAttribute::TANGENTS)
          .build(*_engine);
  auto* spherePositions = new filament::math::float3[_sphereUnitPositions.size()];
  std::copy(_sphereUnitPositions.begin(), _sphereUnitPositions.end(), spherePositions);
  _sphereVertexBuffer->setBufferAt(
      *_engine,
      0,
      filament::VertexBuffer::BufferDescriptor(
          spherePositions,
          _sphereUnitPositions.size() * sizeof(filament::math::float3),
          [](void* buffer, size_t, void*) {
            delete[] static_cast<filament::math::float3*>(buffer);
          }));

  auto* sphereTangents = new filament::math::short4[_sphereUnitTangents.size()];
  std::copy(_sphereUnitTangents.begin(), _sphereUnitTangents.end(), sphereTangents);
  _sphereVertexBuffer->setBufferAt(
      *_engine,
      1,
      filament::VertexBuffer::BufferDescriptor(
          sphereTangents,
          _sphereUnitTangents.size() * sizeof(filament::math::short4),
          [](void* buffer, size_t, void*) {
            delete[] static_cast<filament::math::short4*>(buffer);
          }));
  _sphereIndexBuffer = filament::IndexBuffer::Builder()
                           .indexCount(static_cast<uint32_t>(_sphereUnitIndices.size()))
                           .bufferType(filament::IndexBuffer::IndexType::USHORT)
                           .build(*_engine);
  auto* sphereIndices = new uint16_t[_sphereUnitIndices.size()];
  std::copy(_sphereUnitIndices.begin(), _sphereUnitIndices.end(), sphereIndices);
  _sphereIndexBuffer->setBuffer(
      *_engine,
      filament::IndexBuffer::BufferDescriptor(
          sphereIndices,
          _sphereUnitIndices.size() * sizeof(uint16_t),
          [](void* buffer, size_t, void*) { delete[] static_cast<uint16_t*>(buffer); }));

  CreateSolidSphereBatch();

  // Create solid geometry entity
  _solidEntity = utils::EntityManager::get().create();
  filament::RenderableManager::Builder(1)
      .boundingBox({{-1e6f, -1e6f, -1e6f}, {1e6f, 1e6f, 1e6f}})
      .material(0, _solidMaterial->getDefaultInstance())
      .geometry(
          0,
          filament::RenderableManager::PrimitiveType::TRIANGLES,
          _solidVertexBuffer,
          _solidIndexBuffer,
          0,
          0)
      .culling(false)
      .receiveShadows(false)
      .castShadows(false)
      .geometryType(filament::RenderableManager::Builder::GeometryType::DYNAMIC)
      .build(*_engine, _solidEntity);

  // Create overlay geometry entity
  _overlayEntity = utils::EntityManager::get().create();
  filament::RenderableManager::Builder(1)
      .boundingBox({{-1e6f, -1e6f, -1e6f}, {1e6f, 1e6f, 1e6f}})
      .material(0, _overlayMaterialInstance)
      .geometry(
          0,
          filament::RenderableManager::PrimitiveType::TRIANGLES,
          _overlayVertexBuffer,
          _overlayIndexBuffer,
          0,
          0)
      .culling(false)
      .receiveShadows(false)
      .castShadows(false)
      .geometryType(filament::RenderableManager::Builder::GeometryType::DYNAMIC)
      .build(*_engine, _overlayEntity);
}

DebugDraw::~DebugDraw() {
  auto& entityManager = utils::EntityManager::get();
  for (auto& batch : _sphereBatches) {
    SetSolidSphereBatchSceneMembership(batch, false);
    _engine->destroy(batch.entity);
    _engine->destroy(batch.materialInstance);
    _engine->destroy(batch.dataTexture);
    entityManager.destroy(batch.entity);
  }

  _engine->destroy(_entity);
  _engine->destroy(_solidEntity);
  _engine->destroy(_overlayEntity);
  entityManager.destroy(_entity);
  entityManager.destroy(_solidEntity);
  entityManager.destroy(_overlayEntity);
  _engine->destroy(_material->getDefaultInstance());
  _engine->destroy(_material);
  _engine->destroy(_sphereMaterial);
  _engine->destroy(_overlayMaterialInstance);
  _engine->destroy(_solidMaterial->getDefaultInstance());
  _engine->destroy(_solidMaterial);
  _engine->destroy(_vertexBuffer);
  _engine->destroy(_indexBuffer);
  _engine->destroy(_sphereVertexBuffer);
  _engine->destroy(_sphereIndexBuffer);
  _engine->destroy(_solidVertexBuffer);
  _engine->destroy(_solidIndexBuffer);
  _engine->destroy(_overlayVertexBuffer);
  _engine->destroy(_overlayIndexBuffer);
}

std::unique_ptr<DebugDraw> DebugDraw::Create(filament::Engine* engine, filament::Scene* scene) {
  MOCHI_ASSERT(engine != nullptr);
  MOCHI_ASSERT(scene != nullptr);
  return std::unique_ptr<DebugDraw>(new DebugDraw(engine, scene));
}

void DebugDraw::BuildOrthonormalBasis(
    filament::math::float3 dir,
    filament::math::float3& outU,
    filament::math::float3& outV) {
  // Find perpendicular vectors to form an orthonormal basis
  dir = normalize(dir);
  filament::math::float3 up = (std::abs(dir.z) < 0.99f) ? filament::math::float3{0.0f, 0.0f, 1.0f}
                                                        : filament::math::float3{1.0f, 0.0f, 0.0f};
  outU = normalize(cross(up, dir));
  outV = cross(dir, outU);
}

void DebugDraw::DrawLine(
    filament::math::float3 worldStart,
    filament::math::float3 worldEnd,
    filament::math::float4 color) {
  auto base = static_cast<uint32_t>(_positions.size());
  _positions.push_back(worldStart);
  _positions.push_back(worldEnd);
  _colors.push_back(color);
  _colors.push_back(color);
  _indices.push_back(base);
  _indices.push_back(base + 1);
  _dirty = true;
}

void DebugDraw::DrawSphere(
    filament::math::float3 worldPos,
    float radius,
    filament::math::float4 color,
    int segments) {
  constexpr float k2Pi = 2.0f * std::numbers::pi_v<float>;

  for (int i = 0; i < segments; ++i) {
    float a0 = k2Pi * static_cast<float>(i) / static_cast<float>(segments);
    float a1 = k2Pi * static_cast<float>(i + 1) / static_cast<float>(segments);
    float c0 = std::cos(a0), s0 = std::sin(a0);
    float c1 = std::cos(a1), s1 = std::sin(a1);
    DrawLine(
        worldPos + filament::math::float3{c0 * radius, s0 * radius, 0.0f},
        worldPos + filament::math::float3{c1 * radius, s1 * radius, 0.0f},
        color);
    DrawLine(
        worldPos + filament::math::float3{c0 * radius, 0.0f, s0 * radius},
        worldPos + filament::math::float3{c1 * radius, 0.0f, s1 * radius},
        color);
    DrawLine(
        worldPos + filament::math::float3{0.0f, c0 * radius, s0 * radius},
        worldPos + filament::math::float3{0.0f, c1 * radius, s1 * radius},
        color);
  }
}

void DebugDraw::DrawSolidSphere(
    filament::math::float3 center,
    float radius,
    filament::math::float4 color) {
  _sphereData.push_back({center, radius});
  _sphereData.push_back(color);
  _sphereDirty = true;
}

void DebugDraw::DrawBox(filament::Box const& box, filament::math::float4 color) {
  auto center = box.center;
  auto half = box.halfExtent;
  // 8 corners of the box
  filament::math::float3 corners[8] = {
      center + filament::math::float3{-half.x, -half.y, -half.z},
      center + filament::math::float3{+half.x, -half.y, -half.z},
      center + filament::math::float3{+half.x, +half.y, -half.z},
      center + filament::math::float3{-half.x, +half.y, -half.z},
      center + filament::math::float3{-half.x, -half.y, +half.z},
      center + filament::math::float3{+half.x, -half.y, +half.z},
      center + filament::math::float3{+half.x, +half.y, +half.z},
      center + filament::math::float3{-half.x, +half.y, +half.z},
  };

  // 12 edges of the box
  // Bottom face
  DrawLine(corners[0], corners[1], color);
  DrawLine(corners[1], corners[2], color);
  DrawLine(corners[2], corners[3], color);
  DrawLine(corners[3], corners[0], color);
  // Top face
  DrawLine(corners[4], corners[5], color);
  DrawLine(corners[5], corners[6], color);
  DrawLine(corners[6], corners[7], color);
  DrawLine(corners[7], corners[4], color);
  // Vertical edges
  DrawLine(corners[0], corners[4], color);
  DrawLine(corners[1], corners[5], color);
  DrawLine(corners[2], corners[6], color);
  DrawLine(corners[3], corners[7], color);
}

void DebugDraw::DrawSolidCylinder(
    filament::math::float3 base,
    filament::math::float3 axis,
    float radius,
    filament::math::float4 color,
    int segments,
    bool capped,
    bool overlay) {
  // Select target buffers based on overlay flag
  auto& positions = overlay ? _overlayPositions : _solidPositions;
  auto& normals = overlay ? _overlayNormals : _solidNormals;
  auto& colors = overlay ? _overlayColors : _solidColors;
  auto& indices = overlay ? _overlayIndices : _solidIndices;
  auto& dirty = overlay ? _overlayDirty : _solidDirty;
  constexpr float k2Pi = 2.0f * std::numbers::pi_v<float>;

  filament::math::float3 axisDir = normalize(axis);
  filament::math::float3 u, v;
  BuildOrthonormalBasis(axisDir, u, v);

  filament::math::float3 top = base + axis;

  auto baseIndex = static_cast<uint32_t>(positions.size());

  // Generate vertices for the cylinder body
  // Ring at bottom and ring at top
  for (int i = 0; i <= segments; ++i) {
    float angle = k2Pi * static_cast<float>(i) / static_cast<float>(segments);
    float c = std::cos(angle);
    float s = std::sin(angle);

    filament::math::float3 radialDir = u * c + v * s;
    filament::math::float3 bottomPos = base + radialDir * radius;
    filament::math::float3 topPos = top + radialDir * radius;

    // Bottom vertex
    positions.push_back(bottomPos);
    normals.push_back(radialDir);
    colors.push_back(color);

    // Top vertex
    positions.push_back(topPos);
    normals.push_back(radialDir);
    colors.push_back(color);
  }

  // Generate indices for the cylinder body (quads as 2 triangles)
  // Winding: CCW when viewed from outside (where normal points)
  for (int i = 0; i < segments; ++i) {
    uint32_t b0 = baseIndex + static_cast<uint32_t>(i * 2);
    uint32_t t0 = b0 + 1;
    uint32_t b1 = b0 + 2;
    uint32_t t1 = b0 + 3;

    // Two triangles per quad (CCW from outside)
    indices.push_back(b0);
    indices.push_back(b1);
    indices.push_back(t0);

    indices.push_back(t0);
    indices.push_back(b1);
    indices.push_back(t1);
  }

  if (capped) {
    // Bottom cap
    auto bottomCenterIndex = static_cast<uint32_t>(positions.size());
    positions.push_back(base);
    normals.push_back(-axisDir);
    colors.push_back(color);

    for (int i = 0; i <= segments; ++i) {
      float angle = k2Pi * static_cast<float>(i) / static_cast<float>(segments);
      float c = std::cos(angle);
      float s = std::sin(angle);
      filament::math::float3 radialDir = u * c + v * s;
      positions.push_back(base + radialDir * radius);
      normals.push_back(-axisDir);
      colors.push_back(color);
    }

    for (int i = 0; i < segments; ++i) {
      indices.push_back(bottomCenterIndex);
      indices.push_back(bottomCenterIndex + 1 + static_cast<uint32_t>(i + 1));
      indices.push_back(bottomCenterIndex + 1 + static_cast<uint32_t>(i));
    }

    // Top cap
    auto topCenterIndex = static_cast<uint32_t>(positions.size());
    positions.push_back(top);
    normals.push_back(axisDir);
    colors.push_back(color);

    for (int i = 0; i <= segments; ++i) {
      float angle = k2Pi * static_cast<float>(i) / static_cast<float>(segments);
      float c = std::cos(angle);
      float s = std::sin(angle);
      filament::math::float3 radialDir = u * c + v * s;
      positions.push_back(top + radialDir * radius);
      normals.push_back(axisDir);
      colors.push_back(color);
    }

    for (int i = 0; i < segments; ++i) {
      indices.push_back(topCenterIndex);
      indices.push_back(topCenterIndex + 1 + static_cast<uint32_t>(i));
      indices.push_back(topCenterIndex + 1 + static_cast<uint32_t>(i + 1));
    }
  }

  dirty = true;
}

void DebugDraw::DrawSolidAxisAlignedBox(
    filament::Box const& box,
    filament::math::float4 color,
    bool overlay) {
  // Select target buffers based on overlay flag
  auto& positions = overlay ? _overlayPositions : _solidPositions;
  auto& normals = overlay ? _overlayNormals : _solidNormals;
  auto& colors = overlay ? _overlayColors : _solidColors;
  auto& indices = overlay ? _overlayIndices : _solidIndices;
  auto& dirty = overlay ? _overlayDirty : _solidDirty;

  auto center = box.center;
  auto half = box.halfExtent;

  // 8 corners of the box
  filament::math::float3 corners[8] = {
      center + filament::math::float3{-half.x, -half.y, -half.z}, // 0
      center + filament::math::float3{+half.x, -half.y, -half.z}, // 1
      center + filament::math::float3{+half.x, +half.y, -half.z}, // 2
      center + filament::math::float3{-half.x, +half.y, -half.z}, // 3
      center + filament::math::float3{-half.x, -half.y, +half.z}, // 4
      center + filament::math::float3{+half.x, -half.y, +half.z}, // 5
      center + filament::math::float3{+half.x, +half.y, +half.z}, // 6
      center + filament::math::float3{-half.x, +half.y, +half.z}, // 7
  };

  // Face normals
  filament::math::float3 faceNormals[6] = {
      {0.0f, 0.0f, -1.0f}, // Front  (-Z)
      {0.0f, 0.0f, +1.0f}, // Back   (+Z)
      {-1.0f, 0.0f, 0.0f}, // Left   (-X)
      {+1.0f, 0.0f, 0.0f}, // Right  (+X)
      {0.0f, -1.0f, 0.0f}, // Bottom (-Y)
      {0.0f, +1.0f, 0.0f}, // Top    (+Y)
  };

  // Face vertex indices (CCW winding when looking at front face)
  int faceIndices[6][4] = {
      {0, 3, 2, 1}, // Front  (-Z)
      {5, 6, 7, 4}, // Back   (+Z)
      {4, 7, 3, 0}, // Left   (-X)
      {1, 2, 6, 5}, // Right  (+X)
      {4, 0, 1, 5}, // Bottom (-Y)
      {3, 7, 6, 2}, // Top    (+Y)
  };

  auto baseIndex = static_cast<uint32_t>(positions.size());

  // Add vertices for each face (4 vertices per face, 6 faces = 24 vertices)
  for (int face = 0; face < 6; ++face) {
    for (int v = 0; v < 4; ++v) {
      positions.push_back(corners[faceIndices[face][v]]);
      normals.push_back(faceNormals[face]);
      colors.push_back(color);
    }
  }

  // Add indices for each face (2 triangles per face)
  for (int face = 0; face < 6; ++face) {
    uint32_t faceBase = baseIndex + static_cast<uint32_t>(face * 4);
    // Triangle 1
    indices.push_back(faceBase + 0);
    indices.push_back(faceBase + 1);
    indices.push_back(faceBase + 2);
    // Triangle 2
    indices.push_back(faceBase + 0);
    indices.push_back(faceBase + 2);
    indices.push_back(faceBase + 3);
  }

  dirty = true;
}

void DebugDraw::DrawSolidOrientedBox(
    filament::math::float3 center,
    filament::math::float3 axisX,
    filament::math::float3 axisY,
    filament::math::float3 axisZ,
    filament::math::float4 color,
    bool overlay) {
  // Select target buffers based on overlay flag
  auto& positions = overlay ? _overlayPositions : _solidPositions;
  auto& normals = overlay ? _overlayNormals : _solidNormals;
  auto& colors = overlay ? _overlayColors : _solidColors;
  auto& indices = overlay ? _overlayIndices : _solidIndices;
  auto& dirty = overlay ? _overlayDirty : _solidDirty;

  // 8 corners of the oriented box
  // Each axis vector represents the half-extent in that direction
  filament::math::float3 corners[8] = {
      center - axisX - axisY - axisZ, // 0: ---
      center + axisX - axisY - axisZ, // 1: +--
      center + axisX + axisY - axisZ, // 2: ++-
      center - axisX + axisY - axisZ, // 3: -+-
      center - axisX - axisY + axisZ, // 4: --+
      center + axisX - axisY + axisZ, // 5: +-+
      center + axisX + axisY + axisZ, // 6: +++
      center - axisX + axisY + axisZ, // 7: -++
  };

  // Compute face normals from the axis directions
  filament::math::float3 normX = normalize(axisX);
  filament::math::float3 normY = normalize(axisY);
  filament::math::float3 normZ = normalize(axisZ);

  filament::math::float3 faceNormals[6] = {
      -normZ, // Front  (-Z)
      normZ, // Back   (+Z)
      -normX, // Left   (-X)
      normX, // Right  (+X)
      -normY, // Bottom (-Y)
      normY, // Top    (+Y)
  };

  // Face vertex indices (CCW winding when looking at front face)
  int faceIndices[6][4] = {
      {0, 3, 2, 1}, // Front  (-Z)
      {5, 6, 7, 4}, // Back   (+Z)
      {4, 7, 3, 0}, // Left   (-X)
      {1, 2, 6, 5}, // Right  (+X)
      {4, 0, 1, 5}, // Bottom (-Y)
      {3, 7, 6, 2}, // Top    (+Y)
  };

  auto baseIndex = static_cast<uint32_t>(positions.size());

  // Add vertices for each face (4 vertices per face, 6 faces = 24 vertices)
  for (int face = 0; face < 6; ++face) {
    for (int v = 0; v < 4; ++v) {
      positions.push_back(corners[faceIndices[face][v]]);
      normals.push_back(faceNormals[face]);
      colors.push_back(color);
    }
  }

  // Add indices for each face (2 triangles per face)
  for (int face = 0; face < 6; ++face) {
    uint32_t faceBase = baseIndex + static_cast<uint32_t>(face * 4);
    // Triangle 1
    indices.push_back(faceBase + 0);
    indices.push_back(faceBase + 1);
    indices.push_back(faceBase + 2);
    // Triangle 2
    indices.push_back(faceBase + 0);
    indices.push_back(faceBase + 2);
    indices.push_back(faceBase + 3);
  }

  dirty = true;
}

void DebugDraw::DrawSolidArc(
    filament::math::float3 center,
    filament::math::float3 normal,
    filament::math::float3 startDir,
    float angleRadians,
    float innerRadius,
    float outerRadius,
    filament::math::float4 color,
    int segments,
    bool overlay) {
  MOCHI_ASSERT(segments > 0, "DrawSolidArc requires a positive segment count.");

  // Select target buffers based on overlay flag
  auto& positions = overlay ? _overlayPositions : _solidPositions;
  auto& normals = overlay ? _overlayNormals : _solidNormals;
  auto& colors = overlay ? _overlayColors : _solidColors;
  auto& indices = overlay ? _overlayIndices : _solidIndices;
  auto& dirty = overlay ? _overlayDirty : _solidDirty;

  filament::math::float3 n = normalize(normal);
  filament::math::float3 u = normalize(startDir - n * dot(startDir, n)); // Project onto plane
  filament::math::float3 v = cross(n, u);

  auto baseIndex = static_cast<uint32_t>(positions.size());

  // Generate vertices along the arc
  for (int i = 0; i <= segments; ++i) {
    float t = static_cast<float>(i) / static_cast<float>(segments);
    float angle = t * angleRadians;
    float c = std::cos(angle);
    float s = std::sin(angle);

    filament::math::float3 dir = u * c + v * s;

    // Inner vertex
    positions.push_back(center + dir * innerRadius);
    normals.push_back(n);
    colors.push_back(color);

    // Outer vertex
    positions.push_back(center + dir * outerRadius);
    normals.push_back(n);
    colors.push_back(color);
  }

  // Generate indices for front face (quads between inner and outer rings)
  for (int i = 0; i < segments; ++i) {
    uint32_t i0 = baseIndex + static_cast<uint32_t>(i * 2); // inner current
    uint32_t o0 = i0 + 1; // outer current
    uint32_t i1 = baseIndex + static_cast<uint32_t>((i + 1) * 2); // inner next
    uint32_t o1 = i1 + 1; // outer next

    // Two triangles per quad (CCW from front/normal side)
    indices.push_back(i0);
    indices.push_back(i1);
    indices.push_back(o0);

    indices.push_back(o0);
    indices.push_back(i1);
    indices.push_back(o1);
  }

  // Generate back face vertices (same positions, opposite normal)
  auto backBaseIndex = static_cast<uint32_t>(positions.size());
  for (int i = 0; i <= segments; ++i) {
    float t = static_cast<float>(i) / static_cast<float>(segments);
    float angle = t * angleRadians;
    float c = std::cos(angle);
    float s = std::sin(angle);

    filament::math::float3 dir = u * c + v * s;

    // Inner vertex (back face)
    positions.push_back(center + dir * innerRadius);
    normals.push_back(-n);
    colors.push_back(color);

    // Outer vertex (back face)
    positions.push_back(center + dir * outerRadius);
    normals.push_back(-n);
    colors.push_back(color);
  }

  // Generate indices for back face (reversed winding)
  for (int i = 0; i < segments; ++i) {
    uint32_t i0 = backBaseIndex + static_cast<uint32_t>(i * 2);
    uint32_t o0 = i0 + 1;
    uint32_t i1 = backBaseIndex + static_cast<uint32_t>((i + 1) * 2);
    uint32_t o1 = i1 + 1;

    // Two triangles per quad (CW from front = CCW from back)
    indices.push_back(i0);
    indices.push_back(o0);
    indices.push_back(i1);

    indices.push_back(i1);
    indices.push_back(o0);
    indices.push_back(o1);
  }

  dirty = true;
}

void DebugDraw::DrawSolidCone(
    filament::math::float3 base,
    filament::math::float3 axis,
    float radius,
    filament::math::float4 color,
    int segments,
    bool capped,
    bool overlay) {
  // Select target buffers based on overlay flag
  auto& positions = overlay ? _overlayPositions : _solidPositions;
  auto& normals = overlay ? _overlayNormals : _solidNormals;
  auto& colors = overlay ? _overlayColors : _solidColors;
  auto& indices = overlay ? _overlayIndices : _solidIndices;
  auto& dirty = overlay ? _overlayDirty : _solidDirty;

  constexpr float k2Pi = 2.0f * std::numbers::pi_v<float>;

  filament::math::float3 axisDir = normalize(axis);
  filament::math::float3 u, v;
  BuildOrthonormalBasis(axisDir, u, v);

  filament::math::float3 apex = base + axis;
  float height = length(axis);

  // Calculate the slope angle for proper normals
  float slopeAngle = std::atan2(radius, height);
  float cosSlope = std::cos(slopeAngle);
  float sinSlope = std::sin(slopeAngle);

  auto baseIndex = static_cast<uint32_t>(positions.size());

  // Add apex vertex for the cone body (will be duplicated for each triangle for proper normals)
  // Add base ring vertices and create triangles to apex
  for (int i = 0; i < segments; ++i) {
    float angle0 = k2Pi * static_cast<float>(i) / static_cast<float>(segments);
    float angle1 = k2Pi * static_cast<float>(i + 1) / static_cast<float>(segments);

    float c0 = std::cos(angle0), s0 = std::sin(angle0);
    float c1 = std::cos(angle1), s1 = std::sin(angle1);

    filament::math::float3 radialDir0 = u * c0 + v * s0;
    filament::math::float3 radialDir1 = u * c1 + v * s1;

    // Normal calculation: the normal is perpendicular to the surface
    // It points outward along the radial direction, tilted up by the slope angle
    filament::math::float3 normal0 = radialDir0 * cosSlope + axisDir * sinSlope;
    filament::math::float3 normal1 = radialDir1 * cosSlope + axisDir * sinSlope;
    filament::math::float3 normalMid =
        normalize((radialDir0 + radialDir1) * 0.5f) * cosSlope + axisDir * sinSlope;

    filament::math::float3 basePos0 = base + radialDir0 * radius;
    filament::math::float3 basePos1 = base + radialDir1 * radius;

    // Add triangle: apex, basePos0, basePos1
    positions.push_back(apex);
    normals.push_back(normalMid);
    colors.push_back(color);

    positions.push_back(basePos0);
    normals.push_back(normal0);
    colors.push_back(color);

    positions.push_back(basePos1);
    normals.push_back(normal1);
    colors.push_back(color);
  }

  // Add indices for cone body
  for (int i = 0; i < segments; ++i) {
    uint32_t idx = baseIndex + static_cast<uint32_t>(i * 3);
    indices.push_back(idx);
    indices.push_back(idx + 1);
    indices.push_back(idx + 2);
  }

  if (capped) {
    // Base cap
    auto capCenterIndex = static_cast<uint32_t>(positions.size());
    positions.push_back(base);
    normals.push_back(-axisDir);
    colors.push_back(color);

    for (int i = 0; i <= segments; ++i) {
      float angle = k2Pi * static_cast<float>(i) / static_cast<float>(segments);
      float c = std::cos(angle);
      float s = std::sin(angle);
      filament::math::float3 radialDir = u * c + v * s;
      positions.push_back(base + radialDir * radius);
      normals.push_back(-axisDir);
      colors.push_back(color);
    }

    for (int i = 0; i < segments; ++i) {
      indices.push_back(capCenterIndex);
      indices.push_back(capCenterIndex + 1 + static_cast<uint32_t>(i + 1));
      indices.push_back(capCenterIndex + 1 + static_cast<uint32_t>(i));
    }
  }

  dirty = true;
}

void DebugDraw::DrawSolidSphere(
    filament::math::float3 center,
    float radius,
    filament::math::float4 color,
    int stacks,
    int slices,
    bool overlay) {
  // Select target buffers based on overlay flag
  auto& positions = overlay ? _overlayPositions : _solidPositions;
  auto& normals = overlay ? _overlayNormals : _solidNormals;
  auto& colors = overlay ? _overlayColors : _solidColors;
  auto& indices = overlay ? _overlayIndices : _solidIndices;
  auto& dirty = overlay ? _overlayDirty : _solidDirty;

  constexpr float kPi = std::numbers::pi_v<float>;
  constexpr float k2Pi = 2.0f * kPi;

  auto baseIndex = static_cast<uint32_t>(positions.size());

  // Generate vertices
  for (int stack = 0; stack <= stacks; ++stack) {
    float phi = kPi * static_cast<float>(stack) / static_cast<float>(stacks);
    float sinPhi = std::sin(phi);
    float cosPhi = std::cos(phi);

    for (int slice = 0; slice <= slices; ++slice) {
      float theta = k2Pi * static_cast<float>(slice) / static_cast<float>(slices);
      float sinTheta = std::sin(theta);
      float cosTheta = std::cos(theta);

      filament::math::float3 normal{sinPhi * cosTheta, cosPhi, sinPhi * sinTheta};

      positions.push_back(center + normal * radius);
      normals.push_back(normal);
      colors.push_back(color);
    }
  }

  // Generate indices (CCW winding when viewed from outside)
  for (int stack = 0; stack < stacks; ++stack) {
    for (int slice = 0; slice < slices; ++slice) {
      uint32_t current = baseIndex + static_cast<uint32_t>(stack * (slices + 1) + slice);
      uint32_t next = current + static_cast<uint32_t>(slices + 1);

      // First triangle (CCW from outside)
      indices.push_back(current);
      indices.push_back(current + 1);
      indices.push_back(next);

      // Second triangle (CCW from outside)
      indices.push_back(current + 1);
      indices.push_back(next + 1);
      indices.push_back(next);
    }
  }

  dirty = true;
}

void DebugDraw::EnableLineDepthTest(bool enabled) {
  // The wireframe line renderable uses the line material's default instance.
  filament::MaterialInstance* mi = _material->getDefaultInstance();
  mi->setDepthCulling(enabled);
  mi->setDepthWrite(enabled); // Must also write depth when depth test is enabled
}

void DebugDraw::Commit(std::optional<filament::math::float3> overlaySortViewPos) {
  _overlaySortViewPos = overlaySortViewPos;
  if (_dirty) {
    _dirty = false;
    UpdateBuffers();
  }
  if (_sphereDirty) {
    _sphereDirty = false;
    UpdateSolidSphereInstances();
  }
  if (_solidDirty) {
    _solidDirty = false;
    UpdateSolidBuffers();
  }
  if (_overlayDirty) {
    _overlayDirty = false;
    UpdateOverlayBuffers();
  }
}

void DebugDraw::Clear() {
  if (!_positions.empty()) {
    _positions.clear();
    _colors.clear();
    _indices.clear();
    _dirty = false;
    UpdateBuffers();
  }
  _sphereData.clear();
  _sphereDirty = false;
  UpdateSolidSphereInstances();
  if (!_solidPositions.empty()) {
    _solidPositions.clear();
    _solidNormals.clear();
    _solidColors.clear();
    _solidIndices.clear();
    _solidDirty = false;
    UpdateSolidBuffers();
  }
  if (!_overlayPositions.empty()) {
    _overlayPositions.clear();
    _overlayNormals.clear();
    _overlayColors.clear();
    _overlayIndices.clear();
    _overlayDirty = false;
    UpdateOverlayBuffers();
  }
}

utils::Entity DebugDraw::GetEntity() const {
  return _entity;
}

size_t DebugDraw::GetSolidSphereBatchCount() const {
  return _sphereBatches.size();
}

utils::Entity DebugDraw::GetSolidSphereBatchEntity(size_t batchIndex) const {
  MOCHI_ASSERT(batchIndex < _sphereBatches.size());
  return _sphereBatches[batchIndex].entity;
}

utils::Entity DebugDraw::GetSolidEntity() const {
  return _solidEntity;
}

utils::Entity DebugDraw::GetOverlayEntity() const {
  return _overlayEntity;
}

void DebugDraw::RebuildBuffers(uint32_t requiredVertices, uint32_t requiredIndices) {
  auto& rcm = _engine->getRenderableManager();

  if (requiredVertices > _bufferMaxVertices) {
    _engine->destroy(_vertexBuffer);
    _bufferMaxVertices = requiredVertices * 2;
    _vertexBuffer =
        filament::VertexBuffer::Builder()
            .vertexCount(_bufferMaxVertices)
            .bufferCount(2)
            .attribute(
                filament::VertexAttribute::POSITION,
                0,
                filament::VertexBuffer::AttributeType::FLOAT3)
            .attribute(
                filament::VertexAttribute::COLOR, 1, filament::VertexBuffer::AttributeType::FLOAT4)
            .build(*_engine);
  }

  if (requiredIndices > _bufferMaxIndices) {
    _engine->destroy(_indexBuffer);
    _bufferMaxIndices = requiredIndices * 2;
    _indexBuffer = filament::IndexBuffer::Builder()
                       .indexCount(_bufferMaxIndices)
                       .bufferType(filament::IndexBuffer::IndexType::UINT)
                       .build(*_engine);
  }

  auto ri = rcm.getInstance(_entity);
  rcm.setGeometryAt(
      ri, 0, filament::RenderableManager::PrimitiveType::LINES, _vertexBuffer, _indexBuffer, 0, 0);
}

void DebugDraw::UpdateBuffers() {
  auto& rcm = _engine->getRenderableManager();
  auto ri = rcm.getInstance(_entity);

  auto indexCount = static_cast<uint32_t>(_indices.size());

  if (indexCount == 0) {
    rcm.setGeometryAt(
        ri,
        0,
        filament::RenderableManager::PrimitiveType::LINES,
        _vertexBuffer,
        _indexBuffer,
        0,
        0);
    return;
  }

  auto vertexCount = static_cast<uint32_t>(_positions.size());

  RebuildBuffers(vertexCount, indexCount);
  ri = rcm.getInstance(_entity);

  _vertexBuffer->setBufferAt(
      *_engine,
      0,
      filament::VertexBuffer::BufferDescriptor(
          _positions.data(), vertexCount * sizeof(filament::math::float3)));

  _vertexBuffer->setBufferAt(
      *_engine,
      1,
      filament::VertexBuffer::BufferDescriptor(
          _colors.data(), vertexCount * sizeof(filament::math::float4)));

  _indexBuffer->setBuffer(
      *_engine,
      filament::IndexBuffer::BufferDescriptor(_indices.data(), indexCount * sizeof(uint32_t)));

  rcm.setGeometryAt(
      ri,
      0,
      filament::RenderableManager::PrimitiveType::LINES,
      _vertexBuffer,
      _indexBuffer,
      0,
      indexCount);
}

void DebugDraw::CreateSolidSphereBatch() {
  SphereBatch& batch = _sphereBatches.emplace_back();
  batch.entity = utils::EntityManager::get().create();
  batch.materialInstance = _sphereMaterial->createInstance();
  RebuildSolidSphereInstances(batch, 1);
}

void DebugDraw::RebuildSolidSphereInstances(
    SphereBatch& batch,
    uint32_t requiredInstances,
    bool shrinkRenderable) {
  requiredInstances = std::clamp(requiredInstances, 1u, kMaxSphereInstancesPerBatch);
  if (!shrinkRenderable && requiredInstances <= batch.drawInstanceCount) {
    return;
  }

  uint32_t newDrawInstanceCount = 1;
  while (newDrawInstanceCount < requiredInstances) {
    newDrawInstanceCount *= 2;
  }
  newDrawInstanceCount = std::min(newDrawInstanceCount, kMaxSphereInstancesPerBatch);

  bool const wasInScene = _scene->hasEntity(batch.entity);
  SetSolidSphereBatchSceneMembership(batch, false);
  if (batch.drawInstanceCount > 0) {
    _engine->getRenderableManager().destroy(batch.entity);
  }

  if (newDrawInstanceCount > batch.instanceCapacity) {
    uint32_t const texelCapacity = newDrawInstanceCount * kSphereDataTexelsPerInstance;
    uint32_t const textureHeight =
        (texelCapacity + kSphereDataTextureWidth - 1) / kSphereDataTextureWidth;
    filament::Texture* const oldDataTexture = batch.dataTexture;
    batch.dataTexture = filament::Texture::Builder()
                            .width(kSphereDataTextureWidth)
                            .height(textureHeight)
                            .levels(1)
                            .sampler(filament::Texture::Sampler::SAMPLER_2D)
                            .format(filament::Texture::InternalFormat::RGBA32F)
                            .usage(filament::Texture::Usage::DEFAULT)
                            .build(*_engine);

    filament::TextureSampler const sampler(
        filament::TextureSampler::MinFilter::NEAREST, filament::TextureSampler::MagFilter::NEAREST);
    batch.materialInstance->setParameter("sphereData", batch.dataTexture, sampler);
    batch.materialInstance->setParameter(
        "sphereDataTextureWidth", static_cast<int32_t>(kSphereDataTextureWidth));
    batch.instanceCapacity = newDrawInstanceCount;
    _engine->destroy(oldDataTexture);
  }
  batch.materialInstance->setParameter("activeCount", 0);

  filament::RenderableManager::Builder(1)
      .boundingBox({{-1e6f, -1e6f, -1e6f}, {1e6f, 1e6f, 1e6f}})
      .material(0, batch.materialInstance)
      .geometry(
          0,
          filament::RenderableManager::PrimitiveType::TRIANGLES,
          _sphereVertexBuffer,
          _sphereIndexBuffer)
      .instances(newDrawInstanceCount)
      .culling(false)
      .receiveShadows(false)
      .castShadows(false)
      .geometryType(filament::RenderableManager::Builder::GeometryType::STATIC)
      .build(*_engine, batch.entity);
  batch.drawInstanceCount = newDrawInstanceCount;
  SetSolidSphereBatchSceneMembership(batch, wasInScene);
}

void DebugDraw::SetSolidSphereBatchSceneMembership(SphereBatch const& batch, bool shouldBeInScene) {
  bool const isInScene = _scene->hasEntity(batch.entity);
  if (shouldBeInScene && !isInScene) {
    _scene->addEntity(batch.entity);
  } else if (!shouldBeInScene && isInScene) {
    _scene->remove(batch.entity);
  }
}

void DebugDraw::UpdateSolidSphereInstances() {
  size_t const sphereCount = _sphereData.size() / kSphereDataTexelsPerInstance;
  size_t const activeBatchCount =
      (sphereCount + kMaxSphereInstancesPerBatch - 1) / kMaxSphereInstancesPerBatch;
  while (_sphereBatches.size() < activeBatchCount) {
    CreateSolidSphereBatch();
  }

  for (size_t batchIndex = 0; batchIndex < _sphereBatches.size(); ++batchIndex) {
    SphereBatch& batch = _sphereBatches[batchIndex];
    if (batchIndex >= activeBatchCount) {
      batch.materialInstance->setParameter("activeCount", 0);
      SetSolidSphereBatchSceneMembership(batch, false);
      continue;
    }

    size_t const sphereOffset = batchIndex * kMaxSphereInstancesPerBatch;
    size_t const batchSphereCount =
        std::min(sphereCount - sphereOffset, static_cast<size_t>(kMaxSphereInstancesPerBatch));
    bool const shrinkRenderable =
        batch.drawInstanceCount > 1 && batchSphereCount * 4 <= batch.drawInstanceCount;
    RebuildSolidSphereInstances(batch, static_cast<uint32_t>(batchSphereCount), shrinkRenderable);
    batch.materialInstance->setParameter("activeCount", static_cast<int32_t>(batchSphereCount));

    size_t const texelOffset = sphereOffset * kSphereDataTexelsPerInstance;
    size_t const texelCount = batchSphereCount * kSphereDataTexelsPerInstance;
    auto const uploadHeight =
        static_cast<uint32_t>((texelCount + kSphereDataTextureWidth - 1) / kSphereDataTextureWidth);
    size_t const uploadTexelCount = static_cast<size_t>(kSphereDataTextureWidth) * uploadHeight;
    // The shader cannot sample the padding after the active sphere data.
    auto* uploadData = new filament::math::float4[uploadTexelCount]{};
    std::copy_n(_sphereData.data() + texelOffset, texelCount, uploadData);
    batch.dataTexture->setImage(
        *_engine,
        0,
        0,
        0,
        kSphereDataTextureWidth,
        uploadHeight,
        filament::Texture::PixelBufferDescriptor(
            uploadData,
            uploadTexelCount * sizeof(filament::math::float4),
            filament::Texture::Format::RGBA,
            filament::Texture::Type::FLOAT,
            [](void* buffer, size_t, void*) {
              delete[] static_cast<filament::math::float4*>(buffer);
            }));
    SetSolidSphereBatchSceneMembership(batch, true);
  }
}

void DebugDraw::RebuildSolidBuffers(uint32_t requiredVertices, uint32_t requiredIndices) {
  auto& rcm = _engine->getRenderableManager();

  if (requiredVertices > _solidBufferMaxVertices) {
    _engine->destroy(_solidVertexBuffer);
    _solidBufferMaxVertices = requiredVertices * 2;
    _solidVertexBuffer =
        filament::VertexBuffer::Builder()
            .vertexCount(_solidBufferMaxVertices)
            .bufferCount(3)
            .attribute(
                filament::VertexAttribute::POSITION,
                0,
                filament::VertexBuffer::AttributeType::FLOAT3)
            .attribute(
                filament::VertexAttribute::TANGENTS,
                1,
                filament::VertexBuffer::AttributeType::SHORT4)
            .normalized(filament::VertexAttribute::TANGENTS)
            .attribute(
                filament::VertexAttribute::COLOR, 2, filament::VertexBuffer::AttributeType::FLOAT4)
            .build(*_engine);
  }

  if (requiredIndices > _solidBufferMaxIndices) {
    _engine->destroy(_solidIndexBuffer);
    _solidBufferMaxIndices = requiredIndices * 2;
    _solidIndexBuffer = filament::IndexBuffer::Builder()
                            .indexCount(_solidBufferMaxIndices)
                            .bufferType(filament::IndexBuffer::IndexType::UINT)
                            .build(*_engine);
  }

  auto ri = rcm.getInstance(_solidEntity);
  rcm.setGeometryAt(
      ri,
      0,
      filament::RenderableManager::PrimitiveType::TRIANGLES,
      _solidVertexBuffer,
      _solidIndexBuffer,
      0,
      0);
}

void DebugDraw::UpdateSolidBuffers() {
  auto& rcm = _engine->getRenderableManager();
  auto ri = rcm.getInstance(_solidEntity);

  auto indexCount = static_cast<uint32_t>(_solidIndices.size());

  if (indexCount == 0) {
    rcm.setGeometryAt(
        ri,
        0,
        filament::RenderableManager::PrimitiveType::TRIANGLES,
        _solidVertexBuffer,
        _solidIndexBuffer,
        0,
        0);
    return;
  }

  auto vertexCount = static_cast<uint32_t>(_solidPositions.size());

  RebuildSolidBuffers(vertexCount, indexCount);
  ri = rcm.getInstance(_solidEntity);

  _solidVertexBuffer->setBufferAt(
      *_engine,
      0,
      filament::VertexBuffer::BufferDescriptor(
          _solidPositions.data(), vertexCount * sizeof(filament::math::float3)));

  auto* solidTangents = PackNormalTangentFrames(_solidNormals);
  _solidVertexBuffer->setBufferAt(
      *_engine,
      1,
      filament::VertexBuffer::BufferDescriptor(
          solidTangents,
          vertexCount * sizeof(filament::math::short4),
          [](void* buffer, size_t, void*) {
            delete[] static_cast<filament::math::short4*>(buffer);
          }));

  _solidVertexBuffer->setBufferAt(
      *_engine,
      2,
      filament::VertexBuffer::BufferDescriptor(
          _solidColors.data(), vertexCount * sizeof(filament::math::float4)));

  _solidIndexBuffer->setBuffer(
      *_engine,
      filament::IndexBuffer::BufferDescriptor(_solidIndices.data(), indexCount * sizeof(uint32_t)));

  rcm.setGeometryAt(
      ri,
      0,
      filament::RenderableManager::PrimitiveType::TRIANGLES,
      _solidVertexBuffer,
      _solidIndexBuffer,
      0,
      indexCount);
}

void DebugDraw::RebuildOverlayBuffers(uint32_t requiredVertices, uint32_t requiredIndices) {
  auto& rcm = _engine->getRenderableManager();

  if (requiredVertices > _overlayBufferMaxVertices) {
    _engine->destroy(_overlayVertexBuffer);
    _overlayBufferMaxVertices = requiredVertices * 2;
    _overlayVertexBuffer =
        filament::VertexBuffer::Builder()
            .vertexCount(_overlayBufferMaxVertices)
            .bufferCount(3)
            .attribute(
                filament::VertexAttribute::POSITION,
                0,
                filament::VertexBuffer::AttributeType::FLOAT3)
            .attribute(
                filament::VertexAttribute::TANGENTS,
                1,
                filament::VertexBuffer::AttributeType::SHORT4)
            .normalized(filament::VertexAttribute::TANGENTS)
            .attribute(
                filament::VertexAttribute::COLOR, 2, filament::VertexBuffer::AttributeType::FLOAT4)
            .build(*_engine);
  }

  if (requiredIndices > _overlayBufferMaxIndices) {
    _engine->destroy(_overlayIndexBuffer);
    _overlayBufferMaxIndices = requiredIndices * 2;
    _overlayIndexBuffer = filament::IndexBuffer::Builder()
                              .indexCount(_overlayBufferMaxIndices)
                              .bufferType(filament::IndexBuffer::IndexType::UINT)
                              .build(*_engine);
  }

  auto ri = rcm.getInstance(_overlayEntity);
  rcm.setGeometryAt(
      ri,
      0,
      filament::RenderableManager::PrimitiveType::TRIANGLES,
      _overlayVertexBuffer,
      _overlayIndexBuffer,
      0,
      0);
}

void DebugDraw::UpdateOverlayBuffers() {
  auto& rcm = _engine->getRenderableManager();
  auto ri = rcm.getInstance(_overlayEntity);

  auto indexCount = static_cast<uint32_t>(_overlayIndices.size());

  if (indexCount == 0) {
    rcm.setGeometryAt(
        ri,
        0,
        filament::RenderableManager::PrimitiveType::TRIANGLES,
        _overlayVertexBuffer,
        _overlayIndexBuffer,
        0,
        0);
    return;
  }

  // Sort overlay triangles back-to-front so that, with depth testing disabled, nearer overlay
  // geometry paints over farther geometry and translucent overlays blend in the correct order.
  if (_overlaySortViewPos) {
    SortOverlayTrianglesBackToFront(*_overlaySortViewPos);
  }

  auto vertexCount = static_cast<uint32_t>(_overlayPositions.size());

  RebuildOverlayBuffers(vertexCount, indexCount);
  ri = rcm.getInstance(_overlayEntity);

  _overlayVertexBuffer->setBufferAt(
      *_engine,
      0,
      filament::VertexBuffer::BufferDescriptor(
          _overlayPositions.data(), vertexCount * sizeof(filament::math::float3)));

  auto* overlayTangents = PackNormalTangentFrames(_overlayNormals);
  _overlayVertexBuffer->setBufferAt(
      *_engine,
      1,
      filament::VertexBuffer::BufferDescriptor(
          overlayTangents,
          vertexCount * sizeof(filament::math::short4),
          [](void* buffer, size_t, void*) {
            delete[] static_cast<filament::math::short4*>(buffer);
          }));

  _overlayVertexBuffer->setBufferAt(
      *_engine,
      2,
      filament::VertexBuffer::BufferDescriptor(
          _overlayColors.data(), vertexCount * sizeof(filament::math::float4)));

  _overlayIndexBuffer->setBuffer(
      *_engine,
      filament::IndexBuffer::BufferDescriptor(
          _overlayIndices.data(), indexCount * sizeof(uint32_t)));

  rcm.setGeometryAt(
      ri,
      0,
      filament::RenderableManager::PrimitiveType::TRIANGLES,
      _overlayVertexBuffer,
      _overlayIndexBuffer,
      0,
      indexCount);
}

void DebugDraw::SortOverlayTrianglesBackToFront(filament::math::float3 viewPos) {
  size_t const triCount = _overlayIndices.size() / 3;
  if (triCount < 2) {
    return;
  }
  // Squared distance from the viewpoint to each triangle's centroid; sorting by this is a cheap,
  // good-enough painter's order for the small overlay primitives (joint limits, transform axes).
  std::vector<float> distSq(triCount);
  for (size_t t = 0; t < triCount; ++t) {
    filament::math::float3 const& a = _overlayPositions[_overlayIndices[3 * t + 0]];
    filament::math::float3 const& b = _overlayPositions[_overlayIndices[3 * t + 1]];
    filament::math::float3 const& c = _overlayPositions[_overlayIndices[3 * t + 2]];
    filament::math::float3 const d = (a + b + c) * (1.0f / 3.0f) - viewPos;
    distSq[t] = d.x * d.x + d.y * d.y + d.z * d.z;
  }
  std::vector<uint32_t> order(triCount);
  std::iota(order.begin(), order.end(), 0u);
  std::sort(order.begin(), order.end(), [&distSq](uint32_t l, uint32_t r) {
    return distSq[l] > distSq[r]; // farthest first
  });
  std::vector<uint32_t> sorted;
  sorted.reserve(_overlayIndices.size());
  for (uint32_t const t : order) {
    sorted.push_back(_overlayIndices[3 * t + 0]);
    sorted.push_back(_overlayIndices[3 * t + 1]);
    sorted.push_back(_overlayIndices[3 * t + 2]);
  }
  _overlayIndices.swap(sorted);
}

} // namespace mochi_renderer
