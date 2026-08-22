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

#include <mochi_renderer/debug.h>
#include <mochi_renderer/ibl.h>
#include <mochi_renderer/scene_object.h>
#include <mochi_renderer/types.h>

#include <filament/Camera.h>
#include <filament/Engine.h>
#include <filament/Scene.h>
#include <filament/View.h>

#include <math/quat.h>

#include <functional>

#include <utils/unwindows.h> // Clean up Windows macros (near, far, OPAQUE, etc.)

namespace filament {
class ColorGrading;
} // namespace filament

// Renderer is in the global namespace (defined by clients).
class Renderer;

namespace mochi_viewer {
class MochiViewerServer;
} // namespace mochi_viewer

namespace mochi_renderer {

class MochiRenderer;
class ObservationCamera;
class SceneObject;
class Mesh;

enum class CameraMode { Perspective, Orthographic };

// Result of a pick that also reconstructs the world-space (Filament/render space) hit point.
struct PickResult {
  // Resolved public scene object under the cursor, or null if nothing pickable was hit (the
  // renderable may still be an internal-only object, in which case @ref hit is true but this is
  // null).
  SceneObject* object = nullptr;
  // Filament/render-space position of the hit on the object's surface. Only valid when @ref hit.
  filament::math::float3 worldPosition{};
  // True if any renderable was under the cursor (i.e. @ref worldPosition is valid).
  bool hit = false;
};

// A world-space (Filament/render space) ray, used for per-frame drag-plane intersection.
struct WorldRay {
  filament::math::float3 origin{};
  filament::math::float3 direction{}; // normalized
};

class Scene {
 public:
  static std::unique_ptr<Scene> Create(
      filament::Engine* engine,
      SceneViewSettings const& viewSettings = {});
  ~Scene();

  void ApplyViewSettings(SceneViewSettings const& viewSettings) const;

  // Creates a constant (flat) white ambient indirect light of the given @p intensity. Uses a
  // DC-only spherical-harmonics irradiance so it contributes visible ambient even when no
  // IBL/environment is loaded. A subsequent @ref LoadIbl replaces this with the environment's
  // lighting.
  void CreateIndirectLight(float intensity = 30000.0f);

  void CreateSunlight(
      float intensity = 100000.0f,
      filament::math::float3 direction = {0.6f, -1.0f, 0.8f});

  // Set the opacity (darkness) of the ground shadow-catcher plane (range: 0 to 1).
  // No-op if no ground plane exists.
  void SetGroundShadowStrength(float strength);

  void CreateSkybox(filament::math::float4 color = {0.0f, 1.0f, 0.0f, 1.0f});
  void SetSkyboxVisible(bool visible) const;
  bool GetSkyboxVisible() const;
  void SetIbl(IBL* ibl);

  void CreateGroundPlane(float y = 0, filament::math::float3 planeExtent = {10.0f, 0.0f, 10.0f});
  void DestroyGroundPlane();
  void SetGroundPlaneHeight(float y);
  float ComputeGroundPlaneHeight() const;

  void PickSceneObject(float x, float y, std::function<void(SceneObject*)> const& callback);
  // Like @ref PickSceneObject, but also reconstructs the world-space hit point.
  void PickSceneObjectWithPosition(
      float x,
      float y,
      std::function<void(PickResult const&)> const& callback);

  // Builds a world-space ray through a framebuffer pixel (@p pxX, @p pxY in Filament's
  // bottom-origin GL convention). The origin sits on the near plane; direction is normalized.
  WorldRay ScreenPixelToWorldRay(float pxX, float pxY) const;

  DebugDraw* CreateDebugDraw();
  DebugDraw* GetDebugDraw() const;

  // Small near plane (0.003 m) lets the camera get right up to a surface before it clips;
  // Filament's reversed-Z float depth keeps precision fine despite the wide near/far ratio.
  void
  SetViewport(int width, int height, float fov = 45.0f, float near = 0.003f, float far = 100.0f);
  void SetCameraMode(CameraMode mode);
  CameraMode GetCameraMode() const;
  void SetOrthographicHeight(float height);
  float GetOrthographicHeight() const;
  void CameraLookAt(
      filament::math::double3 from,
      filament::math::double3 to,
      filament::math::double3 up = {0, 1, 0});
  void CameraSetTransform(filament::math::double3 position, filament::math::quat rotation);
  void FocusCameraOnSceneObject(SceneObject* object, filament::math::double3 up = {0, 1, 0});
  filament::math::double3 GetCameraPosition() const;
  filament::math::mat4 GetCameraModelMatrix() const;
  filament::math::mat4 GetCameraViewMatrix() const;
  filament::math::mat4 GetCameraProjectionMatrix() const;
  filament::math::quat GetCameraRotation() const;

  bool GetCameraFocusOnSceneObject(
      SceneObject* object,
      filament::math::double3& outFrom,
      filament::math::double3& outTo,
      float& outOrthoHeight) const;
  bool GetCameraFocusOnAllSceneObjects(
      filament::math::double3& outFrom,
      filament::math::double3& outTo,
      float& outOrthoHeight) const;
  bool GetCameraFocusOnSphere(
      filament::math::float3 center,
      float radius,
      filament::math::double3& outFrom,
      filament::math::double3& outTo) const;

  template <typename T>
  T* AddSceneObjectToScene(std::unique_ptr<T> object) {
    if (object == nullptr) {
      return nullptr;
    }
    auto raw = object.get();
    auto entities = object->GetEntities();
    _scene->addEntities(entities.data(), entities.size());
    _objectsPrivate.push_back(std::move(object));
    if (!raw->_internal) {
      _objectsPublic.push_back(raw);
    }
    return raw;
  }
  std::vector<SceneObject*> const& GetSceneObjects() const;
  void DestroySceneObject(SceneObject* object);
  void DestroyAllSceneObjects();

  // Underlying Filament scene and camera, for building an auxiliary view over this scene (e.g. the
  // highlight overlay pass, which shares the main camera so it lines up exactly with the main
  // view).
  filament::Scene* GetFilamentScene() const;
  filament::Camera* GetCamera() const;
  filament::View* GetView() const;

  std::function<void(SceneObject*)> onDestroySceneObject;

 private:
  Scene(filament::Engine* engine, SceneViewSettings const& viewSettings);

  // Resolves a picked Filament renderable back to the public SceneObject it belongs to (or the
  // object it proxies for, when the hit renderable is an internal highlight stand-in). Returns null
  // if the renderable maps to no public object.
  SceneObject* ResolvePickedRenderable(utils::Entity renderable) const;

  friend class ::Renderer;
  friend class MochiRenderer;
  friend class mochi_viewer::MochiViewerServer;
  friend class ObservationCamera;
  filament::Engine* _engine = nullptr;
  filament::Scene* _scene = nullptr;
  filament::View* _view = nullptr;
  utils::Entity _cameraEntity;
  filament::Camera* _camera = nullptr;
  filament::Skybox* _skybox = nullptr;
  filament::IndirectLight* _indirectLight = nullptr;
  mutable filament::ColorGrading* _colorGrading = nullptr;
  utils::Entity _sunlightEntity;
  utils::Entity _groundPlane;
  filament::Material* _groundPlaneMat = nullptr;
  filament::VertexBuffer* _groundPlaneVB = nullptr;
  filament::IndexBuffer* _groundPlaneIB = nullptr;

  std::vector<std::unique_ptr<SceneObject>> _objectsPrivate;
  std::vector<SceneObject*> _objectsPublic;

  std::unique_ptr<DebugDraw> _debugDraw;

  // Camera mode and orthographic projection state
  CameraMode _cameraMode = CameraMode::Perspective;
  float _orthographicHeight = 10.0f;
  int _viewportWidth = 512;
  int _viewportHeight = 512;
  float _fov = 45.0f;
  float _near = 0.003f; // matches the SetViewport default; see note there
  float _far = 100.0f;
};

} // namespace mochi_renderer
