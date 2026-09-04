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

#include <filament/Box.h>
#include <filament/Engine.h>
#include <utils/Entity.h>

#include <math/vec3.h>
#include <math/vec4.h>

#include <cstddef>
#include <optional>
#include <vector>

namespace filament {
class Scene;
}

namespace mochi_renderer {

class DebugDraw {
 public:
  static std::unique_ptr<DebugDraw> Create(filament::Engine* engine, filament::Scene* scene);
  ~DebugDraw();

  // Debug drawing methods
  void DrawLine(
      filament::math::float3 worldStart,
      filament::math::float3 worldEnd,
      filament::math::float4 color);
  void DrawSphere(
      filament::math::float3 worldPos,
      float radius,
      filament::math::float4 color,
      int segments = 32);
  void DrawBox(filament::Box const& box, filament::math::float4 color);

  // Solid (filled) drawing methods
  /// Draws a solid cylinder along an axis direction
  /// @param base Center of the cylinder's bottom cap
  /// @param axis Direction and length of the cylinder (from base to top)
  /// @param radius Radius of the cylinder
  /// @param color RGBA color (alpha < 1 for translucency)
  /// @param segments Number of radial segments (default 24)
  /// @param capped Whether to draw end caps (default true)
  /// @param overlay If true, renders on top of everything (no depth testing)
  void DrawSolidCylinder(
      filament::math::float3 base,
      filament::math::float3 axis,
      float radius,
      filament::math::float4 color,
      int segments = 24,
      bool capped = true,
      bool overlay = false);

  /// Draws a solid box (axis-aligned)
  /// @param box The box to draw (center + halfExtent)
  /// @param color RGBA color (alpha < 1 for translucency)
  /// @param overlay If true, renders on top of everything (no depth testing)
  void DrawSolidAxisAlignedBox(
      filament::Box const& box,
      filament::math::float4 color,
      bool overlay = false);

  /// Draws a solid oriented box
  /// @param center Center of the box in world space
  /// @param axisX Local X axis direction and length (halfExtent.x)
  /// @param axisY Local Y axis direction and length (halfExtent.y)
  /// @param axisZ Local Z axis direction and length (halfExtent.z)
  /// @param color RGBA color (alpha < 1 for translucency)
  /// @param overlay If true, renders on top of everything (no depth testing)
  void DrawSolidOrientedBox(
      filament::math::float3 center,
      filament::math::float3 axisX,
      filament::math::float3 axisY,
      filament::math::float3 axisZ,
      filament::math::float4 color,
      bool overlay = false);

  /// Draws a solid arc (pie/wedge shape) - useful for visualizing joint limits
  /// @param center Center point of the arc
  /// @param normal Normal vector of the arc plane
  /// @param startDir Direction to the start of the arc (will be normalized)
  /// @param angleRadians Angle of the arc in radians (positive = CCW when looking along normal)
  /// @param innerRadius Inner radius (0 for a pie slice)
  /// @param outerRadius Outer radius
  /// @param color RGBA color (alpha < 1 for translucency)
  /// @param segments Number of angular segments (default 24)
  /// @param overlay If true, renders on top of everything (no depth testing)
  void DrawSolidArc(
      filament::math::float3 center,
      filament::math::float3 normal,
      filament::math::float3 startDir,
      float angleRadians,
      float innerRadius,
      float outerRadius,
      filament::math::float4 color,
      int segments = 24,
      bool overlay = false);

  /// Draws a solid cone
  /// @param base Center of the cone's base
  /// @param axis Direction and length from base to apex
  /// @param radius Radius at the base
  /// @param color RGBA color (alpha < 1 for translucency)
  /// @param segments Number of radial segments (default 24)
  /// @param capped Whether to draw the base cap (default true)
  /// @param overlay If true, renders on top of everything (no depth testing)
  void DrawSolidCone(
      filament::math::float3 base,
      filament::math::float3 axis,
      float radius,
      filament::math::float4 color,
      int segments = 24,
      bool capped = true,
      bool overlay = false);

  /// Draws solid spheres using shared instanced geometry.
  /// @param center Center of the sphere
  /// @param radius Radius of the sphere
  /// @param color RGBA color (alpha < 1 for translucency)
  void DrawSolidSphere(filament::math::float3 center, float radius, filament::math::float4 color);

  /// Draws a tessellated solid sphere.
  /// @param center Center of the sphere
  /// @param radius Radius of the sphere
  /// @param color RGBA color (alpha < 1 for translucency)
  /// @param stacks Number of latitude divisions
  /// @param slices Number of longitude divisions
  /// @param overlay If true, renders on top of everything (no depth testing)
  void DrawSolidSphere(
      filament::math::float3 center,
      float radius,
      filament::math::float4 color,
      int stacks,
      int slices,
      bool overlay = false);

  // Enable/disable depth testing for wireframe (line) drawing. When enabled, lines depth-test and
  // depth-write, so they are occluded by closer geometry but (with a small toward-camera bias from
  // the caller) coincident wireframes still draw on top regardless of draw order. Default
  // (disabled) draws lines on top of everything.
  void EnableLineDepthTest(bool enabled);

  // Uploads any dirty debug geometry. If @p overlaySortViewPos is set, the "on top" overlay
  // triangles are sorted back-to-front from that viewpoint before upload, so that with depth
  // testing disabled nearer overlay geometry paints over farther geometry (and translucent overlays
  // blend in the correct order). Pass the camera position each frame.
  void Commit(std::optional<filament::math::float3> overlaySortViewPos = std::nullopt);
  void Clear();
  utils::Entity GetEntity() const;
  utils::Entity GetSolidEntity() const;
  utils::Entity GetOverlayEntity() const;

 private:
  friend struct DebugDrawTestAccess;

  size_t GetSolidSphereBatchCount() const;
  utils::Entity GetSolidSphereBatchEntity(size_t batchIndex) const;
  struct SphereBatch {
    utils::Entity entity;
    filament::MaterialInstance* materialInstance = nullptr;
    filament::Texture* dataTexture = nullptr;
    uint32_t instanceCapacity = 0;
    uint32_t drawInstanceCount = 0;
  };

  DebugDraw(filament::Engine* engine, filament::Scene* scene);
  void RebuildBuffers(uint32_t requiredVertices, uint32_t requiredIndices);
  void UpdateBuffers();
  void CreateSolidSphereBatch();
  void RebuildSolidSphereInstances(
      SphereBatch& batch,
      uint32_t requiredInstances,
      bool shrinkRenderable = false);
  void SetSolidSphereBatchSceneMembership(SphereBatch const& batch, bool shouldBeInScene);
  void UpdateSolidSphereInstances();
  void RebuildSolidBuffers(uint32_t requiredVertices, uint32_t requiredIndices);
  void UpdateSolidBuffers();
  void RebuildOverlayBuffers(uint32_t requiredVertices, uint32_t requiredIndices);
  void UpdateOverlayBuffers();
  // Reorders _overlayIndices so overlay triangles are drawn farthest-first relative to @p viewPos.
  void SortOverlayTrianglesBackToFront(filament::math::float3 viewPos);

  // Helper to build an orthonormal basis from a direction vector
  static void BuildOrthonormalBasis(
      filament::math::float3 dir,
      filament::math::float3& outU,
      filament::math::float3& outV);

  static constexpr uint32_t kInitialMaxVertices = 64 * 1024;
  static constexpr uint32_t kInitialMaxIndices = 64 * 1024;

  filament::Engine* _engine = nullptr;
  filament::Scene* _scene = nullptr;

  // Wireframe
  utils::Entity _entity;
  filament::Material* _material = nullptr;
  filament::VertexBuffer* _vertexBuffer = nullptr;
  filament::IndexBuffer* _indexBuffer = nullptr;
  std::vector<filament::math::float3> _positions;
  std::vector<filament::math::float4> _colors;
  std::vector<uint32_t> _indices;
  uint32_t _bufferMaxVertices = 0;
  uint32_t _bufferMaxIndices = 0;
  bool _dirty = false;

  // Instanced solid sphere rendering. Each sphere uses two RGBA32F texels:
  // center/radius followed by color.
  filament::Material* _sphereMaterial = nullptr;
  filament::VertexBuffer* _sphereVertexBuffer = nullptr;
  filament::IndexBuffer* _sphereIndexBuffer = nullptr;
  std::vector<filament::math::float3> _sphereUnitPositions;
  std::vector<filament::math::short4> _sphereUnitTangents;
  std::vector<uint16_t> _sphereUnitIndices;
  std::vector<filament::math::float4> _sphereData;
  std::vector<SphereBatch> _sphereBatches;
  bool _sphereDirty = false;

  // Solid (triangles) rendering
  utils::Entity _solidEntity;
  filament::Material* _solidMaterial = nullptr;
  filament::VertexBuffer* _solidVertexBuffer = nullptr;
  filament::IndexBuffer* _solidIndexBuffer = nullptr;
  std::vector<filament::math::float3> _solidPositions;
  std::vector<filament::math::float3> _solidNormals;
  std::vector<filament::math::float4> _solidColors;
  std::vector<uint32_t> _solidIndices;
  uint32_t _solidBufferMaxVertices = 0;
  uint32_t _solidBufferMaxIndices = 0;
  bool _solidDirty = false;

  // Overlay (triangles, no depth testing) rendering
  utils::Entity _overlayEntity;
  filament::MaterialInstance* _overlayMaterialInstance = nullptr;
  filament::VertexBuffer* _overlayVertexBuffer = nullptr;
  filament::IndexBuffer* _overlayIndexBuffer = nullptr;
  std::vector<filament::math::float3> _overlayPositions;
  std::vector<filament::math::float3> _overlayNormals;
  std::vector<filament::math::float4> _overlayColors;
  std::vector<uint32_t> _overlayIndices;
  uint32_t _overlayBufferMaxVertices = 0;
  uint32_t _overlayBufferMaxIndices = 0;
  bool _overlayDirty = false;
  // Viewpoint used to sort overlay triangles back-to-front this frame (set by Commit); unset skips
  // sorting.
  std::optional<filament::math::float3> _overlaySortViewPos;
};

} // namespace mochi_renderer
