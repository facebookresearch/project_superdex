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

#include <mochi_renderer/ibl.h>
#include <mochi_renderer/mesh.h>
#include <mochi_renderer/scene.h>
#include <mochi_renderer/scene_object.h>

#include <filament/ColorGrading.h>
#include <filament/IndexBuffer.h>
#include <filament/IndirectLight.h>
#include <filament/LightManager.h>
#include <filament/Material.h>
#include <filament/RenderableManager.h>
#include <filament/Skybox.h>
#include <filament/TransformManager.h>
#include <filament/VertexBuffer.h>
#include <filament/Viewport.h>

#include <utils/EntityManager.h>
#include <utils/Path.h>

#include <math/mat4.h>
#include <math/norm.h>
#include <math/quat.h>
#include <math/vec3.h>
#include <math/vec4.h>

#include <mochi_core/utils/debug.h>

#include "materials.h"
#include "view_settings.h"

#include <iostream>
#include <memory>
#include <numbers>

#include <mochi_renderer/debug.h>

namespace mochi_renderer {

std::unique_ptr<Scene> Scene::Create(
    filament::Engine* engine,
    mochi_renderer::SceneViewSettings const& viewSettings) {
  MOCHI_ASSERT(engine != nullptr);
  return std::unique_ptr<Scene>(new Scene(engine, viewSettings));
}

Scene::Scene(filament::Engine* engine, mochi_renderer::SceneViewSettings const& viewSettings) {
  _engine = engine;
  _scene = engine->createScene();
  _view = engine->createView();
  _view->setScene(_scene);
  _view->setPostProcessingEnabled(false);
  _view->setChannelDepthClearEnabled(0, true);
  // Include transparent renderables when picking. By default Filament writes picking IDs only in
  // the depth/structure prepass, which filters out blended geometry, so transparent links could not
  // be clicked in the viewport (only selectable via the hierarchy). This adds a picking pass that
  // covers both opaque and transparent renderables.
  _view->setTransparentPickingEnabled(true);
  _cameraEntity = utils::EntityManager::get().create();
  _camera = _engine->createCamera(_cameraEntity);
  _camera->lookAt({1.0f, 0.5f, 1.0f}, {0.0f, 0.0f, 0.0f}, {0, 1, 0});
  _view->setCamera(_camera);
  ApplyViewSettings(viewSettings);
}

Scene::~Scene() {
  if (_colorGrading) {
    _engine->destroy(_colorGrading);
  }
  if (_indirectLight) {
    _engine->destroy(_indirectLight);
  }
  if (!_sunlightEntity.isNull()) {
    _scene->remove(_sunlightEntity);
    _engine->destroy(_sunlightEntity);
  }
  DestroyGroundPlane();
  // An IBL set via SetIbl is owned externally (by the ResourceManager) and may be destroyed
  // before this scene, so clear the scene's references to its skybox/indirect light first.
  _scene->setSkybox(nullptr);
  _scene->setIndirectLight(nullptr);
  if (_skybox) {
    _engine->destroy(_skybox);
  }
  if (_debugDraw) {
    utils::Entity debugEntities[] = {
        _debugDraw->GetEntity(),
        _debugDraw->GetSolidEntity(),
        _debugDraw->GetOverlayEntity(),
    };
    _scene->removeEntities(debugEntities, 3);
    _debugDraw.reset();
  }
  _scene->removeAllEntities();
  _objectsPrivate.clear();
  _engine->destroy(_view);
  _engine->destroyCameraComponent(_cameraEntity);
  utils::EntityManager::get().destroy(_cameraEntity);
  _engine->destroy(_scene);
}

void Scene::ApplyViewSettings(mochi_renderer::SceneViewSettings const& viewSettings) const {
  ApplyViewSettingsToView(_engine, _view, _colorGrading, viewSettings);
  // Skybox visibility is Scene-specific (the offscreen ObservationCamera has no skybox).
  SetSkyboxVisible(viewSettings.showSkybox);
}

void Scene::CreateSkybox(filament::math::float4 color) {
  if (_skybox) {
    _scene->setSkybox(nullptr);
    _engine->destroy(_skybox);
  }
  _skybox = filament::Skybox::Builder().color(color).build(*_engine);
  _scene->setSkybox(_skybox);
}

void Scene::CreateIndirectLight(float intensity) {
  if (_indirectLight) {
    _engine->destroy(_indirectLight);
  }
  // A single-band (DC-only) spherical-harmonics irradiance produces a uniform white ambient term.
  // Without it (intensity only), an IndirectLight with no environment contributes nothing.
  filament::math::float3 const sh[1] = {{1.0f, 1.0f, 1.0f}};
  _indirectLight =
      filament::IndirectLight::Builder().irradiance(1, sh).intensity(intensity).build(*_engine);
  _scene->setIndirectLight(_indirectLight);
}

void Scene::CreateSunlight(float intensity, filament::math::float3 direction) {
  if (_sunlightEntity) {
    _engine->destroy(_sunlightEntity);
  }
  _sunlightEntity = utils::EntityManager::get().create();
  // Higher-resolution directional shadow map (2048 vs the 1024 default) for a crisper ground and
  // self shadow. Filament's default LiSPSM fit is kept (it concentrates resolution near the
  // camera): an earlier stable=true experiment traded that crispness for uniform low-res shadows,
  // and the flicker it aimed to fix was actually screen-space AO, addressed via GTAO above.
  filament::LightManager::ShadowOptions shadowOptions;
  shadowOptions.mapSize = 2048;
  filament::LightManager::Builder(filament::LightManager::Type::SUN)
      .color(
          filament::Color::toLinear<filament::ACCURATE>(
              filament::math::float3{0.98f, 0.92f, 0.89f}))
      .intensity(intensity)
      .direction(direction)
      .sunAngularRadius(1.9f)
      .sunHaloSize(10.0f)
      .sunHaloFalloff(80.0f)
      .castShadows(true)
      .shadowOptions(shadowOptions)
      .build(*_engine, _sunlightEntity);
  _scene->addEntity(_sunlightEntity);
}

void Scene::SetGroundShadowStrength(float strength) {
  if (_groundPlaneMat) {
    _groundPlaneMat->setDefaultParameter("strength", strength);
  }
}

void Scene::SetIbl(IBL* ibl) {
  if (ibl == nullptr) {
    // Clear any previously-set IBL skybox/indirect light.
    _scene->setSkybox(nullptr);
    _scene->setIndirectLight(nullptr);
    return;
  }
  _scene->setSkybox(ibl->GetSkybox());
  _scene->setIndirectLight(ibl->GetIndirectLight());
}

void Scene::CreateGroundPlane(float y, filament::math::float3 planeExtent) {
  DestroyGroundPlane();

  auto& em = utils::EntityManager::get();
  _groundPlaneMat = filament::Material::Builder()
                        .package(
                            MOCHI_RENDERER_MATERIALS_GROUNDSHADOW_DATA,
                            MOCHI_RENDERER_MATERIALS_GROUNDSHADOW_SIZE)
                        .build(*_engine);
  _groundPlaneMat->setDefaultParameter("strength", 0.75f);

  static uint32_t const indices[] = {0, 1, 2, 2, 3, 0};

  // Ground plane on the XZ plane (Y-up convention, matches Filament).
  auto* vertices = new filament::math::float3[4]{
      {-planeExtent.x, 0, -planeExtent.z},
      {-planeExtent.x, 0, planeExtent.z},
      {planeExtent.x, 0, planeExtent.z},
      {planeExtent.x, 0, -planeExtent.z},
  };

  // Tangent frame: tangent=+X, bitangent=+Z, normal=+Y.
  filament::math::short4 const tbn = filament::math::packSnorm16(
      filament::math::mat3f::packTangentFrame(
          filament::math::mat3f{
              filament::math::float3{1.0f, 0.0f, 0.0f},
              filament::math::float3{0.0f, 0.0f, 1.0f},
              filament::math::float3{0.0f, 1.0f, 0.0f}})
          .xyzw);

  auto* normals = new filament::math::short4[4]{tbn, tbn, tbn, tbn};

  auto deleteCb = [](void* buffer, size_t, void*) { delete[] static_cast<uint8_t*>(buffer); };

  _groundPlaneVB =
      filament::VertexBuffer::Builder()
          .vertexCount(4)
          .bufferCount(2)
          .attribute(
              filament::VertexAttribute::POSITION, 0, filament::VertexBuffer::AttributeType::FLOAT3)
          .attribute(
              filament::VertexAttribute::TANGENTS, 1, filament::VertexBuffer::AttributeType::SHORT4)
          .normalized(filament::VertexAttribute::TANGENTS)
          .build(*_engine);

  _groundPlaneVB->setBufferAt(
      *_engine,
      0,
      filament::VertexBuffer::BufferDescriptor(
          vertices, 4 * sizeof(filament::math::float3), deleteCb));
  _groundPlaneVB->setBufferAt(
      *_engine,
      1,
      filament::VertexBuffer::BufferDescriptor(
          normals, 4 * sizeof(filament::math::short4), deleteCb));

  _groundPlaneIB = filament::IndexBuffer::Builder().indexCount(6).build(*_engine);

  _groundPlaneIB->setBuffer(
      *_engine,
      filament::IndexBuffer::BufferDescriptor(
          indices, _groundPlaneIB->getIndexCount() * sizeof(uint32_t)));

  _groundPlane = em.create();
  filament::RenderableManager::Builder(1)
      .boundingBox({{}, {planeExtent.x, 1e-4f, planeExtent.z}})
      .material(0, _groundPlaneMat->getDefaultInstance())
      .geometry(
          0,
          filament::RenderableManager::PrimitiveType::TRIANGLES,
          _groundPlaneVB,
          _groundPlaneIB,
          0,
          6)
      .culling(false)
      .receiveShadows(true)
      .castShadows(false)
      .build(*_engine, _groundPlane);

  _scene->addEntity(_groundPlane);

  auto& tcm = _engine->getTransformManager();
  tcm.setTransform(
      tcm.getInstance(_groundPlane),
      filament::math::mat4f::translation(filament::math::float3{0, y, 0}));
}

void Scene::DestroyGroundPlane() {
  if (!_groundPlane.isNull()) {
    _scene->remove(_groundPlane);
    _engine->destroy(_groundPlane);
    _engine->destroy(_groundPlaneMat);
    _engine->destroy(_groundPlaneVB);
    _engine->destroy(_groundPlaneIB);
    // Reset the handles so a subsequent call (CreateGroundPlane calls this first, and so does the
    // destructor) cannot double-free these resources.
    _groundPlane = {};
    _groundPlaneMat = nullptr;
    _groundPlaneVB = nullptr;
    _groundPlaneIB = nullptr;
  }
}

void Scene::SetGroundPlaneHeight(float y) {
  if (_groundPlane.isNull()) {
    return;
  }
  auto& tcm = _engine->getTransformManager();
  tcm.setTransform(
      tcm.getInstance(_groundPlane),
      filament::math::mat4f::translation(filament::math::float3{0, y, 0}));
}

float Scene::ComputeGroundPlaneHeight() const {
  float minY = std::numeric_limits<float>::max();
  bool hasValidBounds = false;
  for (auto object : _objectsPublic) {
    if (object && !object->_internal) {
      filament::Box aabb = object->GetAABB();
      // GetAABB returns Filament/render space coordinates (Y-up)
      float linkMinY = aabb.center.y - aabb.halfExtent.y;
      minY = std::min(minY, linkMinY);
      hasValidBounds = true;
    }
  }
  return hasValidBounds ? minY : 0.0f;
}

void Scene::SetSkyboxVisible(bool visible) const {
  filament::Skybox* skybox = _scene->getSkybox();
  if (skybox) {
    // Exclude the highlight-overlay bit so the skybox is never drawn into the overlay pass (which
    // would fill it and tint the whole composite); it stays visible in the main view.
    skybox->setLayerMask(
        0xff, visible ? static_cast<uint8_t>(0xff & ~kHighlightOverlayLayer) : 0x00);
  }
}

filament::Scene* Scene::GetFilamentScene() const {
  return _scene;
}

filament::Camera* Scene::GetCamera() const {
  return _camera;
}

filament::View* Scene::GetView() const {
  return _view;
}

bool Scene::GetSkyboxVisible() const {
  filament::Skybox* skybox = _scene->getSkybox();
  if (skybox) {
    return skybox->getLayerMask() & 0xff;
  }
  return false;
}

SceneObject* Scene::ResolvePickedRenderable(utils::Entity renderable) const {
  if (renderable.isNull()) {
    return nullptr;
  }
  for (auto* object : _objectsPublic) {
    if (object->_internal) {
      continue;
    }
    for (auto entity : object->GetEntities()) {
      if (entity == renderable) {
        return object;
      }
    }
  }
  // The hit renderable may belong to an internal stand-in (e.g. a highlight clone that hides its
  // base while the object is selected). These live only in _objectsPrivate; resolve them back to
  // the object they proxy for so selected objects remain pickable.
  for (auto const& object : _objectsPrivate) {
    if (object->_pickProxy == nullptr) {
      continue;
    }
    auto renderables = object->GetEntities();
    for (auto entity : renderables) {
      if (entity == renderable) {
        return object->_pickProxy;
      }
    }
  }
  return nullptr;
}

// Per-request state for an asynchronous pick. Filament resolves picks a frame or more later, and
// several can be in flight at once (e.g. a grab pick on press and a selection pick on release), so
// each request owns its callback plus a snapshot of the camera state that rendered the depth buffer
// it will read. Held behind a shared_ptr because View::pick's inline functor storage is only a few
// words, and so the state is released even if the query is dropped without firing.
struct PickRequest {
  std::function<void(PickResult const&)> callback;
  filament::Viewport viewport;
  filament::math::mat4 projection;
  filament::math::mat4 model;
};

// Reconstructs a world-space position from a pick's screen-space @p fragCoords using the camera
// state captured when the pick was issued. Recipe from Filament View.h (PickingQueryResult):
//   clip  = (fragCoords.xy / viewport.wh, fragCoords.z) * 2 - 1
//   view  = inverse(projection) * clip
//   world = model * view
// @p projection must be the rendering projection (reversed-Z) that produced the depth in
// fragCoords.z.
static filament::math::float3 ReconstructWorldFromPick(
    filament::math::float3 fragCoords,
    filament::Viewport const& viewport,
    filament::math::mat4 const& projection,
    filament::math::mat4 const& model) {
  filament::math::double4 const clip{
      (static_cast<double>(fragCoords.x) / static_cast<double>(viewport.width)) * 2.0 - 1.0,
      (static_cast<double>(fragCoords.y) / static_cast<double>(viewport.height)) * 2.0 - 1.0,
      static_cast<double>(fragCoords.z) * 2.0 - 1.0,
      1.0};
  filament::math::double4 viewPos = inverse(projection) * clip;
  viewPos /= viewPos.w;
  filament::math::double4 const world = model * viewPos;
  return filament::math::float3{
      static_cast<float>(world.x), static_cast<float>(world.y), static_cast<float>(world.z)};
}

void Scene::PickSceneObjectWithPosition(
    float x,
    float y,
    std::function<void(PickResult const&)> const& callback) {
  // Snapshot the camera state now rather than reading it in the callback: fragCoords index the
  // depth buffer of the frame this pick was issued against, so unprojecting them with a camera that
  // has since moved (or a resized viewport) would land on the wrong world point.
  //
  // The callback travels with its own request too: a single pending-callback member would be
  // clobbered whenever two picks are in flight, delivering each result to the wrong requester.
  auto const request = std::make_shared<PickRequest>(PickRequest{
      callback, _view->getViewport(), _camera->getProjectionMatrix(), _camera->getModelMatrix()});
  _view->pick(
      static_cast<uint32_t>(x),
      static_cast<uint32_t>(y),
      [this, request](filament::View::PickingQueryResult const& result) {
        PickResult out;
        out.object = ResolvePickedRenderable(result.renderable);
        if (!result.renderable.isNull()) {
          out.hit = true;
          out.worldPosition = ReconstructWorldFromPick(
              result.fragCoords, request->viewport, request->projection, request->model);
        }
        if (request->callback) {
          request->callback(out);
        }
      });
}

void Scene::PickSceneObject(float x, float y, std::function<void(SceneObject*)> const& callback) {
  PickSceneObjectWithPosition(x, y, [callback](PickResult const& result) {
    if (callback) {
      callback(result.object);
    }
  });
}

WorldRay Scene::ScreenPixelToWorldRay(float pxX, float pxY) const {
  filament::Viewport const vp = _view->getViewport();
  double const ndcX = (static_cast<double>(pxX) / static_cast<double>(vp.width)) * 2.0 - 1.0;
  double const ndcY = (static_cast<double>(pxY) / static_cast<double>(vp.height)) * 2.0 - 1.0;
  // Culling projection is a standard finite-far, invertible OpenGL-style projection (NDC z in
  // [-1, 1]); unproject the near (z=-1) and far (z=+1) points and take their difference.
  filament::math::mat4 const invViewProj =
      inverse(GetCameraProjectionMatrix() * GetCameraViewMatrix());
  filament::math::double4 nearP = invViewProj * filament::math::double4{ndcX, ndcY, -1.0, 1.0};
  filament::math::double4 farP = invViewProj * filament::math::double4{ndcX, ndcY, 1.0, 1.0};
  nearP /= nearP.w;
  farP /= farP.w;
  WorldRay ray;
  ray.origin = filament::math::float3{
      static_cast<float>(nearP.x), static_cast<float>(nearP.y), static_cast<float>(nearP.z)};
  ray.direction = normalize(
      filament::math::float3{
          static_cast<float>(farP.x - nearP.x),
          static_cast<float>(farP.y - nearP.y),
          static_cast<float>(farP.z - nearP.z)});
  return ray;
}

DebugDraw* Scene::CreateDebugDraw() {
  if (_debugDraw) {
    return _debugDraw.get();
  }
  _debugDraw = DebugDraw::Create(_engine);
  _scene->addEntity(_debugDraw->GetEntity());
  _scene->addEntity(_debugDraw->GetSolidEntity());
  _scene->addEntity(_debugDraw->GetOverlayEntity());
  return _debugDraw.get();
}

DebugDraw* Scene::GetDebugDraw() const {
  return _debugDraw.get();
}

static void UpdateOrthographicProjection(
    filament::Camera* camera,
    float orthoHeight,
    float aspect,
    float nearPlane,
    float farPlane) {
  float const halfHeight = orthoHeight * 0.5f;
  float const halfWidth = halfHeight * aspect;
  camera->setProjection(
      filament::Camera::Projection::ORTHO,
      -halfWidth,
      halfWidth,
      -halfHeight,
      halfHeight,
      nearPlane,
      farPlane);
}

void Scene::SetViewport(int width, int height, float inFov, float inNear, float inFar) {
  _viewportWidth = width;
  _viewportHeight = height;
  _fov = inFov;
  _near = inNear;
  _far = inFar;
  _view->setViewport(filament::Viewport{0, 0, (uint32_t)width, (uint32_t)height});
  float const aspect = static_cast<float>(width) / static_cast<float>(height);
  if (_cameraMode == CameraMode::Perspective) {
    _camera->setProjection(inFov, aspect, inNear, inFar);
  } else {
    UpdateOrthographicProjection(_camera, _orthographicHeight, aspect, inNear, inFar);
  }
}

void Scene::SetCameraMode(CameraMode mode) {
  if (_cameraMode == mode) {
    return;
  }
  _cameraMode = mode;
  float const aspect = static_cast<float>(_viewportWidth) / static_cast<float>(_viewportHeight);
  if (_cameraMode == CameraMode::Perspective) {
    _camera->setProjection(_fov, aspect, _near, _far);
  } else {
    UpdateOrthographicProjection(_camera, _orthographicHeight, aspect, _near, _far);
  }
}

CameraMode Scene::GetCameraMode() const {
  return _cameraMode;
}

void Scene::SetOrthographicHeight(float height) {
  _orthographicHeight = height;
  if (_cameraMode == CameraMode::Orthographic) {
    float const aspect = static_cast<float>(_viewportWidth) / static_cast<float>(_viewportHeight);
    UpdateOrthographicProjection(_camera, _orthographicHeight, aspect, _near, _far);
  }
}

float Scene::GetOrthographicHeight() const {
  return _orthographicHeight;
}

void Scene::CameraLookAt(
    filament::math::double3 from,
    filament::math::double3 to,
    filament::math::double3 up) {
  _camera->lookAt({from.x, from.y, from.z}, {to.x, to.y, to.z}, {up.x, up.y, up.z});
}

void Scene::CameraSetTransform(filament::math::double3 position, filament::math::quat rotation) {
  filament::math::mat4f model(rotation);
  model[3] = filament::math::float4{position, 1.0f};
  _camera->setModelMatrix(model);
}

void Scene::FocusCameraOnSceneObject(SceneObject* object, filament::math::double3 up) {
  filament::math::double3 newEye;
  filament::math::double3 center;
  float orthoHeight = 10; // Unused but required by API
  if (GetCameraFocusOnSceneObject(object, newEye, center, orthoHeight)) {
    _camera->lookAt(
        {newEye.x, newEye.y, newEye.z}, {center.x, center.y, center.z}, {up.x, up.y, up.z});
  }
}

filament::math::double3 Scene::GetCameraPosition() const {
  return _camera->getPosition();
}

filament::math::mat4 Scene::GetCameraModelMatrix() const {
  return _camera->getModelMatrix();
}

filament::math::mat4 Scene::GetCameraViewMatrix() const {
  return _camera->getViewMatrix();
}

filament::math::mat4 Scene::GetCameraProjectionMatrix() const {
  // Use culling projection (standard finite far plane) rather than the rendering
  // projection (reversed-Z, infinite far) for compatibility with ImGuizmo and
  // other tools that expect a standard OpenGL-style projection matrix.
  return _camera->getCullingProjectionMatrix();
}

filament::math::quat Scene::GetCameraRotation() const {
  return _camera->getModelMatrix().toQuaternion();
}

static float CalculateOrthographicHeightFromBox(filament::Box const& box) {
  float const maxDimension =
      2.0f * std::max({box.halfExtent.x, box.halfExtent.y, box.halfExtent.z});
  float const marginFactor = 1.2f;
  return maxDimension * marginFactor;
}

bool Scene::GetCameraFocusOnSceneObject(
    SceneObject* object,
    filament::math::double3& outFrom,
    filament::math::double3& outTo,
    float& outOrthoHeight) const {
  if (!object) {
    return false;
  }
  auto box = object->GetAABB();
  if (box.isEmpty()) {
    return false;
  }
  outOrthoHeight = CalculateOrthographicHeightFromBox(box);
  return GetCameraFocusOnSphere(box.center, length(box.halfExtent), outFrom, outTo);
}

bool Scene::GetCameraFocusOnAllSceneObjects(
    filament::math::double3& outFrom,
    filament::math::double3& outTo,
    float& outOrthoHeight) const {
  if (_objectsPublic.empty()) {
    return false;
  }
  filament::Box box = _objectsPublic[0]->GetAABB();
  for (size_t i = 1; i < _objectsPublic.size(); i++) {
    box.unionSelf(_objectsPublic[i]->GetAABB());
  }
  if (box.isEmpty()) {
    return false;
  }
  outOrthoHeight = CalculateOrthographicHeightFromBox(box);
  return GetCameraFocusOnSphere(box.center, length(box.halfExtent), outFrom, outTo);
}

bool Scene::GetCameraFocusOnSphere(
    filament::math::float3 center,
    float radius,
    filament::math::double3& outFrom,
    filament::math::double3& outTo) const {
  if (radius <= 0.0f) {
    return false;
  }
  double fovRad = _fov * std::numbers::pi / 180.0;
  double distance = radius / std::sin(fovRad * 0.5);
  filament::math::double3 forward;
  auto rot = GetCameraRotation();
  forward = normalize(rot * filament::math::double3{0.0, 0.0, -1.0});
  auto newEye = center - forward * distance;
  outFrom = newEye;
  outTo = center;
  return true;
}

std::vector<SceneObject*> const& Scene::GetSceneObjects() const {
  return _objectsPublic;
}

void Scene::DestroySceneObject(SceneObject* object) {
  if (object && onDestroySceneObject) {
    onDestroySceneObject(object);
  }
  if (object) {
    auto entities = object->GetEntities();
    _scene->removeEntities(entities.data(), entities.size());
  }
  auto it = std::find(_objectsPublic.begin(), _objectsPublic.end(), object);
  if (it != _objectsPublic.end()) {
    _objectsPublic.erase(it);
  }
  auto it_ptr = std::find_if(
      _objectsPrivate.begin(),
      _objectsPrivate.end(),
      [object](std::unique_ptr<SceneObject> const& e) { return e.get() == object; });
  if (it_ptr != _objectsPrivate.end()) {
    _objectsPrivate.erase(it_ptr);
  }
}

void Scene::DestroyAllSceneObjects() {
  for (auto& object : _objectsPrivate) {
    auto entities = object->GetEntities();
    _scene->removeEntities(entities.data(), entities.size());
  }
  _objectsPrivate.clear();
  _objectsPublic.clear();
}

} // namespace mochi_renderer
