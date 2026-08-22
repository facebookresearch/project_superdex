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

#include <mochi_core/geometry/aabb.h>
#include <mochi_core/utils/color.h>
#include <mochi_core/utils/coordinate_space.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/nd_array_utils.h>
#include <mochi_core/utils/quaternion.h>
#include <mochi_core/utils/span.h>
#include <mochi_core/utils/transform_rt.h>

#include <imgui.h> // for ImTextureID

#include <cstdint>
#include <memory>
#include <unordered_map>

// Forward declarations for mochi_renderer / app-local renderer types, so this header stays free of
// Filament and windowing includes.
namespace filament {
class Engine;
} // namespace filament
namespace mochi_renderer {
class MochiRenderer;
class Scene;
class DebugDraw;
class Mesh;
class MaterialInstance;
class ResourceManager;
} // namespace mochi_renderer
class Renderer; // app-local Filament renderer
class RenderTarget; // app-local offscreen render target

namespace mochi::dbg {

struct Camera;

struct RenderSceneParams {
  bool showDebugDraw = true;
  bool showOriginTriAxis = true;
  bool showMeshes = true;
  float ambientLightIntensity = 0.2f;
  float directionalLightIntensity = 0.8f;
  float directionalLightYawDeg = 25.0f;
  float directionalLightPitchDeg = -75.0f;
  bool dynamicMeshesShareMaterial = false;
  bool staticMeshesShareMaterial = true;
  bool useFlatShading = false;
  // Surface finish (PBR), tunable live. Defaults approximate hard plastic: dielectric (no metal),
  // moderately smooth so it has a defined but not mirror-like highlight.
  float materialRoughness = 0.35f;
  float materialMetallic = 0.0f;
  float materialReflectance = 0.5f;
  Color staticMeshColor = MakeColor(0xA6B0B4FF);
  Color dynamicMeshColor = MakeColor(0xE89F46FF);
};

// Opaque handle to a mesh added via RenderScene::AddMesh. Zero is never a valid id.
using MeshId = uint32_t;
constexpr MeshId kInvalidMeshId = 0;

// Which shared-material bucket a mesh belongs to. The renderer keeps a live-tunable color/material
// per bucket (see RenderSceneParams::static/dynamicMesh*); the caller picks the bucket. This is
// independent of the AddMesh isDynamic flag, which describes whether the mesh geometry deforms.
enum class MeshMaterialClass { Static, Dynamic };

// Rotating per-actor color: HSL with golden-angle hue distribution for good variety. Matches the
// samples app so the debugger viewport looks the same.
Color GetRotatingColor(int index);

// Compute one smooth normal per node as the normalized, area-weighted average of incident face
// normals. Outputs a flat xyz normal per node (same layout as positions). Used for deformable
// (dynamic) meshes, which keep a fixed vertex count so they can be updated in place — unlike
// BuildSmoothMesh, which re-indexes/welds and changes the vertex count.
void ComputeSmoothNormals(
    Span<float const> positions,
    Span<int const> connectivity,
    DynamicArray<float>& outNormals);

// Owns the Filament-backed render scene (lighting, skybox, debug draw) and an offscreen render
// target exposed to ImGui. Meshes are added directly by the host via AddMesh (no physics coupling).
class RenderScene {
 public:
  static std::unique_ptr<RenderScene> Create(mochi_renderer::MochiRenderer* mochiRenderer);
  ~RenderScene();

  RenderScene(RenderScene const&) = delete;
  RenderScene& operator=(RenderScene const&) = delete;
  RenderScene(RenderScene&&) = delete;
  RenderScene& operator=(RenderScene&&) = delete;

  // ImGui texture handle for the offscreen target. Re-fetch after Render() (it changes on resize).
  ImTextureID GetTextureId() const;

  void SetParams(RenderSceneParams const& params);

  // Set the coordinate space that the scene is expressed in. All inputs to and outputs from the
  // RenderScene class will be in this coordinate space.
  void SetCoordinateSpace(CoordinateSpace const& simSpace);

  // Get the coordinate space convention that was set.
  CoordinateSpace GetCoordinateSpace() const {
    return _simSpace;
  }

  // Direct mesh API. positions/normals are flat xyz triples; connectivity is triangle indices.
  // A non-dynamic mesh (isDynamic=false) bakes flat/smooth normals from its connectivity and
  // rebuilds when the flat-shading toggle changes; a dynamic mesh keeps the supplied normals and
  // can be updated per frame via UpdateMeshGeometry. materialClass selects the shared-material
  // bucket used for coloring, independent of isDynamic (which describes whether the geometry
  // deforms). Returns an opaque id (kInvalidMeshId on failure).
  MeshId AddMesh(
      Span<float const> positions,
      Span<float const> normals,
      Span<int const> connectivity,
      TransformRT const& worldFromLocal,
      Color color,
      MeshMaterialClass materialClass = MeshMaterialClass::Dynamic,
      bool isDynamic = false);
  void UpdateMeshTransform(MeshId id, TransformRT const& worldFromLocal);
  // Hide or show a single mesh. A hidden mesh keeps its GPU buffers (so toggling is instant) and is
  // excluded from GetSceneBounds.
  void SetMeshVisible(MeshId id, bool visible);
  void UpdateMeshGeometry(MeshId id, Span<float const> positions, Span<float const> normals);
  void RemoveMesh(MeshId id);
  void ClearMeshes();

  // Debug-draw geometry (from the synced scene). Positions are flat xyz triples; line colors are
  // flat RGBA (4/vertex, uint8), sphere colors flat RGBA (4/sphere), radii one per sphere. Set only
  // when the data changes; IssueDebugDraw re-issues it every frame (debug draw is cleared per
  // frame).
  void SetDebugLines(Span<float const> positions, Span<uint8_t const> colors);
  void
  SetDebugSpheres(Span<float const> positions, Span<float const> radii, Span<uint8_t const> colors);
  void ClearDebugDraw();

  // Union of all mesh world AABBs (recomputed each Render). Used to frame the camera.
  Aabb GetSceneBounds() const {
    return _sceneBounds;
  }
  bool HasSceneBounds() const {
    return _hasSceneBounds;
  }

  // Resize the offscreen target to match the ImGui content region. Must be called during UI build,
  // before drawing the render-target image, so the drawn texture stays valid for the frame.
  // logicalWidth/Height are the ImGui content-region size; fbScale is DisplayFramebufferScale.
  void Resize(int logicalWidth, int logicalHeight, float fbScale);

  // Render one frame into the offscreen target (at its current size). Call after the UI build.
  void Render(Camera const& camera);

 private:
  explicit RenderScene(mochi_renderer::MochiRenderer* mochiRenderer);

  struct MeshState {
    mochi_renderer::Mesh* mesh = nullptr; // owned by _scene
    std::shared_ptr<mochi_renderer::MaterialInstance> material;
    TransformRT worldFromLocal;
    Color color = {};
    bool isDynamic = false;
    // Which shared-material bucket this mesh uses. Drives coloring, independent of isDynamic (which
    // describes whether the mesh geometry deforms).
    MeshMaterialClass materialClass = MeshMaterialClass::Dynamic;
    bool visible = true;
    bool isTransparent = false;
    bool isFlat = false;
    bool meshIsClosed = true;
    Aabb worldBounds;
    bool hasBounds = false;
    // Stored geometry so static meshes can be rebuilt when the flat/smooth toggle changes.
    DynamicArray<float> positions;
    DynamicArray<float> normals;
    DynamicArray<int> connectivity;
  };

  void SetCamera(Camera const& camera, int width, int height);
  void ApplyLighting();
  // Simulation space -> Filament: scale to metres, and mirror Z for a left-handed simulation.
  Real3 FilamentFromSim(Real3 const& point) const {
    return point * _filamentScale;
  }
  real FilamentFromSim(real length) const {
    return length * _metersPerUnit;
  }
  // The same, for a direction: the mirror only, so the result stays unit length.
  Real3 FilamentDirectionFromSim(Real3 const& direction) const {
    return _mirrors ? Real3{direction[0], direction[1], -direction[2]} : direction;
  }
  void ApplyMeshTransform(MeshState const& state) const;
  void BuildMeshObject(MeshState& state);
  void EnsureMeshMaterial(MeshState& state);
  Color ComputeMeshColor(MeshState const& state) const;
  static void UpdateMeshBounds(MeshState& state);
  std::shared_ptr<mochi_renderer::MaterialInstance> CreateMaterial(Color color, bool flat) const;
  void IssueDebugDraw(Float3 cameraPos);

  filament::Engine* _engine = nullptr;
  mochi_renderer::ResourceManager* _resourceManager = nullptr;
  std::unique_ptr<mochi_renderer::Scene> _scene;
  std::unique_ptr<::Renderer> _renderer;
  std::unique_ptr<::RenderTarget> _renderTarget;
  mochi_renderer::DebugDraw* _debugDraw = nullptr; // owned by _scene

  RenderSceneParams _params;

  // Simulation coordinate space, and everything derived from it. _filamentScale is (s, s, +-s)
  // with s = 1/unitsPerMeter; the negative component mirrors a left-handed space into Filament's
  // right-handed world.
  CoordinateSpace _simSpace = CoordinateSpace::Filament();
  Real3 _filamentScale = {1_r, 1_r, 1_r};
  real _metersPerUnit = 1_r;
  bool _mirrors = false;

  std::unordered_map<MeshId, MeshState> _meshes;
  MeshId _nextMeshId = kInvalidMeshId + 1;
  bool _appliedFlatShading = false;

  // Debug-draw geometry, owned so IssueDebugDraw can re-issue it every frame. Lines: 2 verts per
  // segment (positions 3/vert, colors 4/vert RGBA). Spheres: positions 3 each, one radius + one
  // RGBA color each.
  DynamicArray<float> _dbgLinePositions;
  DynamicArray<uint8_t> _dbgLineColors;
  DynamicArray<float> _dbgSpherePositions;
  DynamicArray<float> _dbgSphereRadii;
  DynamicArray<uint8_t> _dbgSphereColors;

  // Cached lighting state, re-applied only when the inputs change (avoids per-frame light
  // rebuilds).
  bool _lightingInitialized = false;
  float _appliedAmbientIntensity = 0.0f;
  float _appliedDirectionalIntensity = 0.0f;
  float _appliedYawDeg = 0.0f;
  float _appliedPitchDeg = 0.0f;

  // Union of mesh world AABBs (from Render); used to frame the orthographic cameras so the whole
  // scene stays within the near/far slab, and returned to the host for camera framing.
  Aabb _sceneBounds;
  bool _hasSceneBounds = false;
};

using RenderScenePtr = std::unique_ptr<RenderScene>;

} // namespace mochi::dbg
