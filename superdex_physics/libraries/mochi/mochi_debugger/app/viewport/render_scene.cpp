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

#include "camera.h"
#include "render_scene.h"
#include "render_target.h"
#include "renderer.h"

#include <mochi_core/geometry/geometry_utils.h>
#include <mochi_core/geometry/triangular_mesh.h>
#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/nd_array_utils.h>
#include <mochi_core/utils/quaternion_utils.h>
#include <mochi_core/utils/span.h>
#include <mochi_renderer/debug.h>
#include <mochi_renderer/material.h>
#include <mochi_renderer/mesh.h>
#include <mochi_renderer/mochi_renderer.h>
#include <mochi_renderer/resource_manager.h>
#include <mochi_renderer/scene.h>
#include <mochi_renderer/scene_object.h>
#include <mochi_renderer/type_conversions.h>

#include <filament/Color.h>
#include <math/quat.h>
#include <math/vec3.h>
#include <math/vec4.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <tuple>

namespace mochi::dbg {

using mochi_renderer::ToFilament;

namespace {

constexpr int kDefaultViewportWidth = 512;
constexpr int kDefaultViewportHeight = 512;

// Map the UI's normalized [0,1] light intensities to Filament's physical units.
constexpr float kSunIntensityScale = 100000.0f; // lux
constexpr float kIndirectIntensityScale = 30000.0f;

constexpr float kMaxAngleDegForSmoothNormals = 80.0f;

// Depth range for the orthographic camera: the near/far slab is padded by a tenth of the scene
// radius, floored so a tiny scene still gets a usable slab, and the radius itself is floored so
// a degenerate scene cannot produce a zero-depth one.
constexpr float kOrthoMarginFloorMeters = 1.0f;
constexpr float kMinSceneRadiusMeters = 1e-3f;

// Temp storage for a mesh that is being re-built.
struct PreprocessedMesh {
  DynamicArray<float> positions; // flat xyz
  DynamicArray<float> normals; // flat xyz
  DynamicArray<int> indices;
};

} // namespace

static filament::math::float3 ToLinear(Color c) {
  filament::math::float3 const srgb{c[0] / 255.0f, c[1] / 255.0f, c[2] / 255.0f};
  return filament::Color::toLinear<filament::ACCURATE>(srgb);
}

// Nudge a debug-draw point slightly toward the camera so coincident wireframes (e.g. a mesh
// outline drawn at the same depth as its surface) win the depth test. Proportional to distance so
// the bias stays roughly constant in screen depth across the scene.
static filament::math::float3
BiasTowardCamera(filament::math::float3 p, filament::math::float3 cameraPos, float fraction) {
  return p + (cameraPos - p) * fraction;
}

static Float3 PositionAt(Span<float const> positions, int node) {
  return {positions[node * 3 + 0], positions[node * 3 + 1], positions[node * 3 + 2]};
}

// Convert a flat RGBA (uint8) color at index i to a float4 in [0,1].
static filament::math::float4 FlatColorAt(DynamicArray<uint8_t> const& colors, size_t i) {
  return {
      colors[i * 4 + 0] / 255.0f,
      colors[i * 4 + 1] / 255.0f,
      colors[i * 4 + 2] / 255.0f,
      colors[i * 4 + 3] / 255.0f};
}

// Fully faceted: each triangle gets its own 3 vertices, all carrying the triangle's face normal.
// Renders flat with the smooth (Lit) material.
static PreprocessedMesh BuildFlatMesh(Span<float const> positions, Span<int const> connectivity) {
  size_t const triCount = connectivity.size() / 3;
  PreprocessedMesh out;
  out.positions.reserve(triCount * 9);
  out.normals.reserve(triCount * 9);
  out.indices.resize_noinit(triCount * 3);
  std::iota(out.indices.begin(), out.indices.end(), 0);
  for (size_t t = 0; t < triCount; ++t) {
    Float3 const p[3] = {
        PositionAt(positions, connectivity[t * 3 + 0]),
        PositionAt(positions, connectivity[t * 3 + 1]),
        PositionAt(positions, connectivity[t * 3 + 2])};
    Float3 const n = Normalize(Cross(p[1] - p[0], p[2] - p[0]));
    for (auto const& v : p) {
      out.positions.append(v);
      out.normals.append(n);
    }
  }
  return out;
}

// Hard-edge-aware normals: a vertex's normal averages only the incident faces within
// smoothingAngleDeg of the current face, so smooth surfaces stay smooth while sharp edges (e.g. a
// cube's 90-degree corners) stay crisp. Corners with the same position and normal are welded.
static PreprocessedMesh BuildSmoothMesh(
    Span<float const> positions,
    Span<int const> connectivity,
    float smoothingAngleDeg) {
  size_t const nodeCount = positions.size() / 3;
  size_t const triCount = connectivity.size() / 3;
  float const cosThreshold = Cos(smoothingAngleDeg * static_cast<float>(kRadiansPerDegree));

  DynamicArray<Float3> faceNormal(triCount);
  DynamicArray<DynamicArray<int>> nodeTris(nodeCount);
  for (size_t t = 0; t < triCount; ++t) {
    Float3 const p0 = PositionAt(positions, connectivity[t * 3 + 0]);
    Float3 const p1 = PositionAt(positions, connectivity[t * 3 + 1]);
    Float3 const p2 = PositionAt(positions, connectivity[t * 3 + 2]);
    faceNormal[t] = Normalize(Cross(p1 - p0, p2 - p0));
    for (int k = 0; k < 3; ++k) {
      nodeTris[connectivity[t * 3 + k]].push_back(static_cast<int>(t));
    }
  }

  PreprocessedMesh out;
  out.indices.resize_noinit(triCount * 3);
  // Weld key: (node index, quantized normal). Quantizing groups corners with near-identical
  // normals.
  std::map<std::tuple<int, int, int, int>, int> weld;
  auto quantize = [](float c) { return static_cast<int>(Round(c * 1000.0f)); };

  for (size_t t = 0; t < triCount; ++t) {
    for (int k = 0; k < 3; ++k) {
      int const node = connectivity[t * 3 + k];
      Float3 normal{};
      for (int const t2 : nodeTris[node]) {
        if (Dot(faceNormal[t], faceNormal[t2]) >= cosThreshold) {
          normal += faceNormal[t2];
        }
      }
      normal = Normalize(normal);
      auto const key =
          std::make_tuple(node, quantize(normal[0]), quantize(normal[1]), quantize(normal[2]));
      auto const [it, inserted] = weld.try_emplace(key, static_cast<int>(out.positions.size() / 3));
      if (inserted) {
        out.positions.append(PositionAt(positions, node));
        out.normals.append(normal);
      }
      out.indices[t * 3 + k] = it->second;
    }
  }
  return out;
}

static bool ComputeMeshClosed(DynamicArray<int> const& connectivity) {
  if (connectivity.empty()) {
    return true;
  }
  return IsMeshClosed(GenerateEdgeToElementsMap(Unflatten<Int3 const>(connectivity)));
}

// Rotating per-actor color: HSL with golden-angle hue distribution for good variety.
Color GetRotatingColor(int index) {
  float const hue = fmodf(index * 137.5f, 360.0f);
  float const saturation = 0.7f;
  float const lightness = 0.6f;

  float const c = (1.f - fabsf(2.f * lightness - 1.f)) * saturation;
  float const x = c * (1.f - fabsf(fmodf(hue / 60.f, 2.f) - 1.f));
  float const m = lightness - c / 2.f;

  float r = 0;
  float g = 0;
  float b = 0;
  if (hue < 60) {
    r = c;
    g = x;
  } else if (hue < 120) {
    r = x;
    g = c;
  } else if (hue < 180) {
    g = c;
    b = x;
  } else if (hue < 240) {
    g = x;
    b = c;
  } else if (hue < 300) {
    r = x;
    b = c;
  } else {
    r = c;
    b = x;
  }

  return {
      static_cast<uint8_t>((r + m) * 255),
      static_cast<uint8_t>((g + m) * 255),
      static_cast<uint8_t>((b + m) * 255),
      static_cast<uint8_t>(255)};
}

void ComputeSmoothNormals(
    Span<float const> positions,
    Span<int const> connectivity,
    DynamicArray<float>& outNormals) {
  outNormals.clear();
  outNormals.resize(positions.size()); // one xyz normal per node, zero-initialized

  size_t const triCount = connectivity.size() / 3;
  for (size_t t = 0; t < triCount; ++t) {
    int const nodes[3] = {
        connectivity[t * 3 + 0], connectivity[t * 3 + 1], connectivity[t * 3 + 2]};
    Float3 const p0 = PositionAt(positions, nodes[0]);
    Float3 const p1 = PositionAt(positions, nodes[1]);
    Float3 const p2 = PositionAt(positions, nodes[2]);
    // Un-normalized cross product; its magnitude is proportional to the triangle area, so summing
    // it area-weights each face's contribution.
    Float3 const faceNormal = Cross(p1 - p0, p2 - p0);
    for (int const node : nodes) {
      outNormals[node * 3 + 0] += faceNormal[0];
      outNormals[node * 3 + 1] += faceNormal[1];
      outNormals[node * 3 + 2] += faceNormal[2];
    }
  }

  size_t const nodeCount = positions.size() / 3;
  for (size_t n = 0; n < nodeCount; ++n) {
    Float3 v{outNormals[n * 3 + 0], outNormals[n * 3 + 1], outNormals[n * 3 + 2]};
    float const len = Norm(v);
    if (len > 1e-20f) {
      v = v / len;
    }
    outNormals[n * 3 + 0] = v[0];
    outNormals[n * 3 + 1] = v[1];
    outNormals[n * 3 + 2] = v[2];
  }
}

std::unique_ptr<RenderScene> RenderScene::Create(mochi_renderer::MochiRenderer* mochiRenderer) {
  MOCHI_ASSERT(mochiRenderer != nullptr);
  return std::unique_ptr<RenderScene>(new RenderScene(mochiRenderer));
}

RenderScene::RenderScene(mochi_renderer::MochiRenderer* mochiRenderer) {
  _engine = mochiRenderer->GetEngine();
  _resourceManager = mochiRenderer->GetResourceManager();

  // Default view settings enable shadows, SSAO, bloom, MSAA and tone mapping. Use a higher MSAA
  // sample count than the default so thin debug-draw lines (1px hardware lines) get enough
  // coverage samples to stay solid without tiny breaks.
  mochi_renderer::SceneViewSettings viewSettings;
  viewSettings.msaaSampleCount = 8;
  _scene = mochi_renderer::Scene::Create(_engine, viewSettings);
  _scene->SetViewport(kDefaultViewportWidth, kDefaultViewportHeight);
  _scene->CreateSkybox();
  _scene->SetSkyboxVisible(false);
  _scene->CreateSunlight();
  _scene->CreateIndirectLight();
  _scene->CreateDebugDraw();
  _debugDraw = _scene->GetDebugDraw();
  // Debug lines should be occluded by closer meshes (with a small toward-camera bias applied per
  // line in IssueDebugDraw so coincident wireframes still draw on top).
  _debugDraw->EnableLineDepthTest(true);

  _renderer = ::Renderer::Create(
      _engine, mochiRenderer->GetFilamentRenderer(), kDefaultViewportWidth, kDefaultViewportHeight);
  _renderer->SetClearColor({0.048f, 0.048f, 0.055f, 1.0f});

  _renderTarget = ::RenderTarget::Create(_engine, kDefaultViewportWidth, kDefaultViewportHeight);
}

RenderScene::~RenderScene() = default;

ImTextureID RenderScene::GetTextureId() const {
  return _renderTarget ? _renderTarget->GetTextureId() : nullptr;
}

void RenderScene::SetParams(RenderSceneParams const& params) {
  _params = params;
}

void RenderScene::SetCoordinateSpace(CoordinateSpace const& simSpace) {
  simSpace.Validate(ErrorAssert{});
  _simSpace = simSpace;
  _metersPerUnit = 1_r / simSpace.unitsPerMeter;

  // Filament is right-handed, so a left-handed simulation has to be mirrored on its way in
  // or every object renders as its own reflection. Which axis is mirrored is arbitrary: any
  // other choice differs by a rigid rotation that the camera receives as well.
  _mirrors = Dot(Cross(simSpace.GetRight(), simSpace.GetUp()), -simSpace.GetForward()) < 0_r;
  _filamentScale = {_metersPerUnit, _metersPerUnit, _mirrors ? -_metersPerUnit : _metersPerUnit};

  // Meshes may already exist, and the sun direction is cached across frames.
  for (auto const& [id, st] : _meshes) {
    ApplyMeshTransform(st);
  }
  _lightingInitialized = false;
}

// Push a mesh's simulation-space transform to Filament. The per-axis scale carries the unit and
// handedness conversion, so the vertex buffer stays in simulation coordinates: a deformable body
// that re-uploads its vertices every frame pays nothing for the conversion.
void RenderScene::ApplyMeshTransform(MeshState const& st) const {
  if (!st.mesh) {
    return;
  }
  // Mirroring a rotation conjugates it, which negates the quaternion components in the
  // mirrored plane; the negative scale component then carries the mirror itself, and
  // Filament reverses the triangle winding and the normal matrix for us.
  Quaternion rotation = st.worldFromLocal.GetRotation();
  if (_mirrors) {
    Real4 const q = rotation.ToReal4();
    rotation = Quaternion{-q[0], -q[1], q[2], q[3]};
  }
  st.mesh->SetLocalTransform(
      rotation,
      FilamentFromSim(st.worldFromLocal.GetTranslation()),
      _filamentScale,
      /*converter*/ nullptr);
}

MeshId RenderScene::AddMesh(
    Span<float const> positions,
    Span<float const> normals,
    Span<int const> connectivity,
    TransformRT const& worldFromLocal,
    Color color,
    MeshMaterialClass materialClass,
    bool isDynamic) {
  if (positions.size() % 3 != 0 || connectivity.size() % 3 != 0) {
    return kInvalidMeshId;
  }
  if (!normals.empty() && normals.size() != positions.size()) {
    return kInvalidMeshId;
  }
  size_t const nodeCount = positions.size() / 3;
  for (int const index : connectivity) {
    if (index < 0 || static_cast<size_t>(index) >= nodeCount) {
      return kInvalidMeshId;
    }
  }
  if (connectivity.empty() || positions.empty()) {
    return kInvalidMeshId;
  }

  MeshId const id = _nextMeshId++;
  MeshState& st = _meshes[id];
  st.isDynamic = isDynamic;
  st.materialClass = materialClass;
  st.color = color;
  st.worldFromLocal = worldFromLocal;
  st.positions = positions;
  st.normals = normals;
  st.connectivity = connectivity;
  st.meshIsClosed = ComputeMeshClosed(st.connectivity);

  BuildMeshObject(st);
  UpdateMeshBounds(st);
  return id;
}

void RenderScene::UpdateMeshTransform(MeshId id, TransformRT const& worldFromLocal) {
  auto it = _meshes.find(id);
  if (it == _meshes.end()) {
    return;
  }
  MeshState& st = it->second;
  st.worldFromLocal = worldFromLocal;
  ApplyMeshTransform(st);
  UpdateMeshBounds(st);
}

void RenderScene::SetMeshVisible(MeshId id, bool visible) {
  auto it = _meshes.find(id);
  if (it == _meshes.end()) {
    return;
  }
  it->second.visible = visible;
}

void RenderScene::UpdateMeshGeometry(
    MeshId id,
    Span<float const> positions,
    Span<float const> normals) {
  auto it = _meshes.find(id);
  if (it == _meshes.end()) {
    return;
  }
  MeshState& st = it->second;
  MOCHI_ASSERT(st.isDynamic, "UpdateMeshGeometry is only valid for dynamic meshes.");
  st.positions = positions;
  st.normals = normals;
  if (st.mesh) {
    st.mesh->Update(st.positions, st.normals);
  }
  UpdateMeshBounds(st);
}

void RenderScene::RemoveMesh(MeshId id) {
  auto it = _meshes.find(id);
  if (it == _meshes.end()) {
    return;
  }
  if (it->second.mesh) {
    _scene->DestroySceneObject(it->second.mesh);
  }
  _meshes.erase(it);
}

void RenderScene::ClearMeshes() {
  for (auto& [id, st] : _meshes) {
    if (st.mesh) {
      _scene->DestroySceneObject(st.mesh);
    }
  }
  _meshes.clear();
}

void RenderScene::SetDebugLines(Span<float const> positions, Span<uint8_t const> colors) {
  MOCHI_ASSERT_VERBOSE(positions.size() % 3 == 0, "Must be a multiple of 3");
  MOCHI_ASSERT_VERBOSE(colors.size() % 4 == 0, "Must be a multiple of 4");
  MOCHI_ASSERT_VERBOSE(positions.size() / 3 == colors.size() / 4, "Size mismatch");
  _dbgLinePositions = positions;
  _dbgLineColors = colors;
}

void RenderScene::SetDebugSpheres(
    Span<float const> positions,
    Span<float const> radii,
    Span<uint8_t const> colors) {
  MOCHI_ASSERT_VERBOSE(positions.size() % 3 == 0, "Must be a multiple of 3");
  MOCHI_ASSERT_VERBOSE(colors.size() % 4 == 0, "Must be a multiple of 4");
  MOCHI_ASSERT_VERBOSE(radii.size() == positions.size() / 3, "Size mismatch");
  MOCHI_ASSERT_VERBOSE(radii.size() == colors.size() / 4, "Size mismatch");
  _dbgSpherePositions = positions;
  _dbgSphereRadii = radii;
  _dbgSphereColors = colors;
}

void RenderScene::ClearDebugDraw() {
  _dbgLinePositions.clear();
  _dbgLineColors.clear();
  _dbgSpherePositions.clear();
  _dbgSphereRadii.clear();
  _dbgSphereColors.clear();
}

void RenderScene::UpdateMeshBounds(MeshState& st) {
  size_t const nodeCount = st.positions.size() / 3;
  if (nodeCount == 0) {
    st.hasBounds = false;
    return;
  }
  Real3 const first =
      st.worldFromLocal.TransformPoint(StaticCast<Real3>(PositionAt(st.positions, 0)));
  Real3 mn = first;
  Real3 mx = first;
  for (size_t i = 1; i < nodeCount; ++i) {
    Real3 const p = st.worldFromLocal.TransformPoint(
        StaticCast<Real3>(PositionAt(st.positions, static_cast<int>(i))));
    for (int c = 0; c < 3; ++c) {
      mn[c] = std::min(mn[c], p[c]);
      mx[c] = std::max(mx[c], p[c]);
    }
  }
  st.worldBounds = Aabb{mn, mx};
  st.hasBounds = true;
}

Color RenderScene::ComputeMeshColor(MeshState const& state) const {
  if (state.materialClass == MeshMaterialClass::Static && _params.staticMeshesShareMaterial) {
    return _params.staticMeshColor;
  }
  if (state.materialClass == MeshMaterialClass::Dynamic && _params.dynamicMeshesShareMaterial) {
    return _params.dynamicMeshColor;
  }
  return state.color;
}

std::shared_ptr<mochi_renderer::MaterialInstance> RenderScene::CreateMaterial(
    Color color,
    bool flat) const {
  filament::math::float3 const base = ToLinear(color);
  if (color[3] < 255) {
    return _resourceManager->CreateLitSeeThroughMaterial(base, color[3] / 255.0f);
  }
  if (flat) {
    return _resourceManager->CreateFlatLitOpaqueMaterial(base);
  }
  return _resourceManager->CreateLitOpaqueMaterial(base);
}

void RenderScene::EnsureMeshMaterial(MeshState& state) {
  Color const desired = ComputeMeshColor(state);
  bool const wantTransparent = desired[3] < 255;
  // Static meshes bake flatness into geometry and always use the smooth material; only dynamic
  // meshes express flat shading via the flat material.
  bool const wantFlatMaterial = _params.useFlatShading && state.isDynamic;

  if (!state.material || state.isTransparent != wantTransparent ||
      state.isFlat != wantFlatMaterial) {
    // Material type changed (or first creation): recreate it.
    state.material = CreateMaterial(desired, wantFlatMaterial);
    state.isTransparent = wantTransparent;
    state.isFlat = wantFlatMaterial;
    state.color = desired;
    if (state.mesh) {
      state.mesh->SetMaterial(state.material);
    }
  } else if (state.color != desired) {
    // Same material type: just recolor (no reallocation).
    auto* mi = state.material->Get();
    mi->setParameter("baseColor", ToLinear(desired));
    if (wantTransparent) {
      mi->setParameter("alpha", desired[3] / 255.0f);
    }
    state.color = desired;
  }

  // Apply the (live-tunable) surface finish each frame. All three lit materials expose these.
  auto* mi = state.material->Get();
  mi->setParameter("roughness", Clamp(_params.materialRoughness, 0.0f, 1.0f));
  mi->setParameter("metallic", Clamp(_params.materialMetallic, 0.0f, 1.0f));
  mi->setParameter("reflectance", Clamp(_params.materialReflectance, 0.0f, 1.0f));
}

void RenderScene::BuildMeshObject(MeshState& st) {
  if (st.mesh) {
    _scene->DestroySceneObject(st.mesh);
    st.mesh = nullptr;
  }

  // EnsureMeshMaterial seeds st.material and the cached color/flags before the mesh is created
  // (CreateMesh requires a material).
  EnsureMeshMaterial(st);

  if (!st.isDynamic) {
    // Static meshes render with the smooth material; flat vs hard-edge look is baked into the mesh
    // normals here (preprocessed once; rebuilt on the flat/smooth toggle).
    PreprocessedMesh const pm = _params.useFlatShading
        ? BuildFlatMesh(st.positions, st.connectivity)
        : BuildSmoothMesh(st.positions, st.connectivity, kMaxAngleDegForSmoothNormals);
    auto mesh = mochi_renderer::Mesh::CreateMesh(
        _engine,
        pm.positions,
        pm.normals,
        pm.indices,
        st.material,
        /*isDynamic*/ false,
        st.meshIsClosed);
    st.mesh = _scene->AddSceneObjectToScene(std::move(mesh));
  } else {
    // Dynamic meshes keep the supplied smooth normals and update per frame; the flat/smooth look
    // comes from the material instead.
    auto mesh = mochi_renderer::Mesh::CreateMesh(
        _engine,
        st.positions,
        st.normals,
        st.connectivity,
        st.material,
        /*isDynamic*/ true,
        st.meshIsClosed);
    st.mesh = _scene->AddSceneObjectToScene(std::move(mesh));
  }

  ApplyMeshTransform(st);
}

void RenderScene::SetCamera(Camera const& camera, int width, int height) {
  Real3 camPos = StaticCast<Real3>(camera.position); // simulation units
  float const fovDeg = camera.verticalFov * static_cast<float>(kDegreesPerRadian);
  // Filament's world stays metric, so its metre-denominated settings (ambient-occlusion
  // radius, shadow bias and near/far hints) keep working; distances scale on the way in.
  auto const toMeters = [this](float length) { return FilamentFromSim(length); };

  if (camera.mode == Camera::Mode::Perspective) {
    _scene->SetCameraMode(mochi_renderer::CameraMode::Perspective);
    _scene->SetViewport(width, height, fovDeg, toMeters(camera.nearZ), toMeters(camera.farZ));
  } else {
    // Orthographic depth: an ortho image does not depend on how far the camera is along its view
    // axis, so place the camera behind the whole scene AABB and set near/far to span it. The
    // scene never clips however large it grows, while the in-plane pan is preserved. (Without
    // this the camera sits at the scene center and the near plane clips half the scene.)
    float nearZ = camera.nearZ;
    float farZ = camera.farZ;
    if (_hasSceneBounds && IsFinite(_sceneBounds.GetMin()) && IsFinite(_sceneBounds.GetMax())) {
      Sphere const sphere = GetBoundingSphere(_sceneBounds);
      Real3 const center = sphere.GetCenter();
      auto const unitsPerMeter = static_cast<float>(_simSpace.unitsPerMeter);
      float const radius =
          std::max(static_cast<float>(sphere.GetRadius()), kMinSceneRadiusMeters * unitsPerMeter);
      float const margin = std::max(kOrthoMarginFloorMeters * unitsPerMeter, radius * 0.1f);
      float const back = radius + 2.0f * margin;
      // Unit (rotation of a unit vector), in simulation coordinates like camPos and center.
      auto const fwd = StaticCast<Real3>(camera.GetForward());
      Real3 const d = camPos - center;
      auto const along = static_cast<float>(Dot(d, fwd));
      camPos = camPos - fwd * static_cast<real>(along + back);
      nearZ = margin;
      farZ = 2.0f * radius + 3.0f * margin;
    }
    // mochi_renderer derives the ortho bounds from the height and viewport aspect
    // (halfHeight = orthoHeight/2, halfWidth = halfHeight * aspect), matching the previous manual
    // projection. Set the mode and height first; SetViewport then applies the near/far slab.
    _scene->SetCameraMode(mochi_renderer::CameraMode::Orthographic);
    _scene->SetOrthographicHeight(toMeters(camera.orthoHeight));
    _scene->SetViewport(width, height, fovDeg, toMeters(nearZ), toMeters(farZ));
  }

  // Filament's camera looks down its own -Z with +X right and +Y up, so its rotation is the
  // matrix whose columns are the camera's right, up and backward axes, mirrored into
  // Filament. Mirroring cancels a left-handed space's handedness flip, leaving a proper
  // rotation.
  Real3 const camRight = FilamentDirectionFromSim(StaticCast<Real3>(camera.GetRight()));
  Real3 const camUp = FilamentDirectionFromSim(StaticCast<Real3>(camera.GetUp()));
  Real3 const camBackward = FilamentDirectionFromSim(StaticCast<Real3>(-camera.GetForward()));
  Matrix3x3r cameraBasis{};
  for (size_t row = 0; row < 3; ++row) {
    cameraBasis[row][0] = camRight[row];
    cameraBasis[row][1] = camUp[row];
    cameraBasis[row][2] = camBackward[row];
  }
  Real4 const q = QuaternionFromMatrix(cameraBasis).ToReal4(); // (x, y, z, w)
  filament::math::quat fq;
  fq.x = q[0];
  fq.y = q[1];
  fq.z = q[2];
  fq.w = q[3];
  Real3 const filamentPos = FilamentFromSim(camPos);
  _scene->CameraSetTransform({filamentPos[0], filamentPos[1], filamentPos[2]}, fq);
}

void RenderScene::ApplyLighting() {
  bool const changed = !_lightingInitialized ||
      _params.ambientLightIntensity != _appliedAmbientIntensity ||
      _params.directionalLightIntensity != _appliedDirectionalIntensity ||
      _params.directionalLightYawDeg != _appliedYawDeg ||
      _params.directionalLightPitchDeg != _appliedPitchDeg;
  if (!changed) {
    return;
  }

  // The sun is aimed with the same yaw/pitch convention as the camera, so the lighting angle
  // -- and the meaning of the UI sliders -- is the same in every simulation space.
  float const yaw = _params.directionalLightYawDeg * static_cast<float>(kRadiansPerDegree);
  float const pitch = _params.directionalLightPitchDeg * static_cast<float>(kRadiansPerDegree);
  Real3 const dir = StaticCast<Real3>(Camera::DirectionFromYawPitch(_simSpace, yaw, pitch));

  _scene->CreateSunlight(
      _params.directionalLightIntensity * kSunIntensityScale,
      ToFilament(FilamentDirectionFromSim(dir)));
  // Flat white ambient; the slider drives its intensity.
  _scene->CreateIndirectLight(_params.ambientLightIntensity * kIndirectIntensityScale);

  _lightingInitialized = true;
  _appliedAmbientIntensity = _params.ambientLightIntensity;
  _appliedDirectionalIntensity = _params.directionalLightIntensity;
  _appliedYawDeg = _params.directionalLightYawDeg;
  _appliedPitchDeg = _params.directionalLightPitchDeg;
}

void RenderScene::IssueDebugDraw(Float3 cameraPos) {
  filament::math::float3 const camPos = ToFilament(FilamentFromSim(StaticCast<Real3>(cameraPos)));
  // Small toward-camera depth bias so coincident wireframes (e.g. a mesh outline drawn at the same
  // depth as its surface) win the depth test, while lines genuinely behind geometry stay occluded.
  constexpr float kDepthBias = 0.0025f;
  // DebugDraw takes world-space points and has no model matrix of its own, so every primitive
  // input is converted here. That is O(primitives), not O(vertices).
  auto const toView = [&](Float3 const& p) {
    return BiasTowardCamera(ToFilament(FilamentFromSim(p)), camPos, kDepthBias);
  };

  // Origin tri-axis (RGB = the simulation's own XYZ).
  if (_params.showOriginTriAxis) {
    real const axisLen = 0.1_r * _simSpace.unitsPerMeter;
    filament::math::float3 const origin = toView({0_r, 0_r, 0_r});
    _debugDraw->DrawLine(origin, toView({axisLen, 0_r, 0_r}), {1, 0, 0, 1});
    _debugDraw->DrawLine(origin, toView({0_r, axisLen, 0_r}), {0, 1, 0, 1});
    _debugDraw->DrawLine(origin, toView({0_r, 0_r, axisLen}), {0, 0, 1, 1});
  }

  if (_params.showDebugDraw) {
    // Synced debug lines: 2 vertices per segment. DebugDraw takes one color per line, so use the
    // first endpoint's color (visually negligible for debug viz).
    size_t const lineVertCount = _dbgLinePositions.size() / 3;
    for (size_t i = 0; i + 1 < lineVertCount; i += 2) {
      filament::math::float4 const color = (i * 4 + 3) < _dbgLineColors.size()
          ? FlatColorAt(_dbgLineColors, i)
          : filament::math::float4{1, 1, 1, 1};
      _debugDraw->DrawLine(
          toView(PositionAt(_dbgLinePositions, i)),
          toView(PositionAt(_dbgLinePositions, i + 1)),
          color);
    }

    // Synced debug spheres (wireframe).
    for (size_t i = 0; i < _dbgSphereRadii.size(); ++i) {
      _debugDraw->DrawSphere(
          toView(PositionAt(_dbgSpherePositions, i)),
          FilamentFromSim(_dbgSphereRadii[i]),
          FlatColorAt(_dbgSphereColors, i));
    }
  }
}

void RenderScene::Resize(int logicalWidth, int logicalHeight, float fbScale) {
  int const renderWidth = std::max(1, static_cast<int>(logicalWidth * fbScale));
  int const renderHeight = std::max(1, static_cast<int>(logicalHeight * fbScale));
  _renderTarget->Resize(renderWidth, renderHeight); // no-op if unchanged
}

void RenderScene::Render(Camera const& camera) {
  int renderWidth = 0;
  int renderHeight = 0;
  _renderTarget->GetSize(renderWidth, renderHeight);

  // Recompute the union of mesh world bounds (used for ortho framing and returned to the host).
  _hasSceneBounds = false;
  for (auto& [id, st] : _meshes) {
    if (!st.hasBounds || !st.visible) {
      continue;
    }
    _sceneBounds = _hasSceneBounds ? GetAabb(_sceneBounds, st.worldBounds) : st.worldBounds;
    _hasSceneBounds = true;
  }

  SetCamera(camera, renderWidth, renderHeight);
  ApplyLighting();

  // The flat/smooth toggle changes baked normals of static meshes, so rebuild them when it flips.
  if (_appliedFlatShading != _params.useFlatShading) {
    for (auto& [id, st] : _meshes) {
      if (!st.isDynamic) {
        BuildMeshObject(st);
      }
    }
    _appliedFlatShading = _params.useFlatShading;
  }

  // Update per-mesh materials/colors and visibility.
  for (auto& [id, st] : _meshes) {
    if (!st.mesh) {
      continue;
    }
    if (_params.showMeshes) {
      EnsureMeshMaterial(st);
    }
    st.mesh->SetVisible(_params.showMeshes && st.visible);
  }

  // Debug draw lifecycle: issue draws -> commit -> render -> clear.
  IssueDebugDraw(camera.position);
  _debugDraw->Commit();
  _renderer->Render(_scene.get(), _renderTarget.get(), /*flushAndWait*/ true);
  _debugDraw->Clear();
}

} // namespace mochi::dbg
