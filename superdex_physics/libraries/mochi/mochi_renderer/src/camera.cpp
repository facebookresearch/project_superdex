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

#include <mochi_renderer/camera.h>
#include <mochi_renderer/scene.h>

#include "offscreen_render_target.h"

#include "materials.h"
#include "view_settings.h"

#include <filament/ColorGrading.h>
#include <filament/IndexBuffer.h>
#include <filament/Material.h>
#include <filament/RenderableManager.h>
#include <filament/TextureSampler.h>
#include <filament/VertexBuffer.h>
#include <filament/Viewport.h>
#include <utils/EntityManager.h>

#include <mochi_core/utils/debug.h>

#include <algorithm>
#include <cmath>
#include <future>

namespace mochi_renderer {

namespace {

// Filament's Vulkan backend produces readPixels data that is 180° rotated for offscreen
// render targets. Compensate by negating both X and Y in the projection matrix (a rotation,
// not a reflection, so winding order is preserved and culling is unaffected).
filament::math::mat4 FlipProjectionXY(filament::math::mat4 proj) {
  proj[0][0] = -proj[0][0];
  proj[1][1] = -proj[1][1];
  proj[2][0] = -proj[2][0];
  proj[2][1] = -proj[2][1];
  return proj;
}

} // namespace

ObservationCamera::ObservationCamera(
    filament::Engine* engine,
    Scene* scene,
    filament::Renderer* renderer,
    Config const& config)
    : _engine(engine), _scene(scene), _renderer(renderer), _config(config) {
  MOCHI_ASSERT(engine != nullptr);
  MOCHI_ASSERT(scene != nullptr);

  // Create camera entity
  _cameraEntity = utils::EntityManager::get().create();
  _camera = _engine->createCamera(_cameraEntity);
  _camera->lookAt({0, -1, 0.5}, {0, 0, 0}, {0, 0, 1});

  // Create view pointing at the scene
  _view = _engine->createView();
  _view->setScene(scene->_scene);
  _view->setCamera(_camera);

  // Apply basic view settings to match the scene's view configuration.
  // This ensures proper rendering with tone mapping and post-processing.
  _view->setPostProcessingEnabled(true);
  _view->setShadowingEnabled(true);
  _view->setShadowType(filament::View::ShadowType::PCF);

  // Create offscreen render target
  _renderTarget = OffscreenRenderTarget::Create(_engine, _config.width, _config.height);

  // Set up the view to render to the offscreen target
  _view->setRenderTarget(_renderTarget->GetFilamentRenderTarget());
  _view->setViewport(
      filament::Viewport{
          0, 0, static_cast<uint32_t>(_config.width), static_cast<uint32_t>(_config.height)});

  // Create a headless swap chain once (reused across frames).
  _swapChain = _engine->createSwapChain(
      static_cast<uint32_t>(_config.width), static_cast<uint32_t>(_config.height));

  UpdateProjection();
}

ObservationCamera::~ObservationCamera() {
  DestroyDistortionPass();
  if (_colorGrading) {
    _engine->destroy(_colorGrading);
  }
  if (_view) {
    _engine->destroy(_view);
  }
  _renderTarget.reset();
  if (_swapChain) {
    _engine->destroy(_swapChain);
  }
  if (_camera) {
    _engine->destroyCameraComponent(_cameraEntity);
    utils::EntityManager::get().destroy(_cameraEntity);
  }
}

void ObservationCamera::SetTransform(
    filament::math::double3 position,
    filament::math::quat rotation) {
  _localPosition = position;
  _localRotation = rotation;
  if (!IsAttached()) {
    filament::math::mat4f model(rotation);
    model[3] = filament::math::float4{position, 1.0f};
    _camera->setModelMatrix(model);
  }
}

void ObservationCamera::LookAt(filament::math::double3 eye, filament::math::double3 target) {
  _camera->lookAt(eye, target, {0, 0, 1});
  _localPosition = eye;
  _localRotation = _camera->getModelMatrix().toQuaternion();
}

filament::math::double3 ObservationCamera::GetPosition() const {
  return _camera->getPosition();
}

filament::math::quat ObservationCamera::GetRotation() const {
  return _camera->getModelMatrix().toQuaternion();
}

void ObservationCamera::SetResolution(int width, int height) {
  _config.width = width;
  _config.height = height;
  _renderTarget->Resize(width, height);
  _view->setRenderTarget(_renderTarget->GetFilamentRenderTarget());

  // Recreate the headless swap chain at the new resolution.
  if (_swapChain) {
    _engine->destroy(_swapChain);
  }
  _swapChain =
      _engine->createSwapChain(static_cast<uint32_t>(width), static_cast<uint32_t>(height));
  _view->setViewport(
      filament::Viewport{0, 0, static_cast<uint32_t>(width), static_cast<uint32_t>(height)});
  if (_distortionTarget) {
    _distortionTarget->Resize(width, height);
    _distortionView->setRenderTarget(_distortionTarget->GetFilamentRenderTarget());
    _distortionView->setViewport(
        filament::Viewport{0, 0, static_cast<uint32_t>(width), static_cast<uint32_t>(height)});
    if (_distortionMaterialInstance) {
      _distortionMaterialInstance->setParameter(
          "imageSize",
          filament::math::float2{static_cast<float>(width), static_cast<float>(height)});
      // Re-bind the source texture after resize
      filament::TextureSampler sampler(
          filament::TextureSampler::MinFilter::LINEAR, filament::TextureSampler::MagFilter::LINEAR);
      _distortionMaterialInstance->setParameter(
          "sourceTexture", _renderTarget->GetColorTexture(), sampler);
    }
  }
  UpdateProjection();
}

void ObservationCamera::SetFov(float degrees) {
  _config.fovDegrees = degrees;
  UpdateProjection();
}

void ObservationCamera::SetClipPlanes(float nearPlane, float farPlane) {
  _config.nearPlane = nearPlane;
  _config.farPlane = farPlane;
  UpdateProjection();
}

void ObservationCamera::ApplyViewSettings(SceneViewSettings const& settings) {
  ApplyViewSettingsToView(_engine, _view, _colorGrading, settings);
}

void ObservationCamera::AttachToObject(std::string const& objectName) {
  _attachedObjectName = objectName;
}

void ObservationCamera::Detach() {
  _attachedObjectName.clear();
  // Apply the local transform directly
  SetTransform(_localPosition, _localRotation);
}

void ObservationCamera::RenderFrame() {
  if (IsAttached()) {
    UpdateAttachedTransform();
  }
  if (_renderer->beginFrame(_swapChain)) {
    _renderer->render(_view);
    // If distortion is active, run the distortion pass
    if (_distortionView) {
      _renderer->render(_distortionView);
    }
    _renderer->endFrame();
  }
}

RenderResult ObservationCamera::ReadPixels() {
  RenderResult result;
  result.width = _config.width;
  result.height = _config.height;
  result.channels = 4;
  result.format = ImageFormat::RGBA8;
  result.pixels.resize(static_cast<size_t>(_config.width) * _config.height * result.channels);

  std::promise<void> readbackPromise;
  auto readbackFuture = readbackPromise.get_future();

  auto callback = [](void* /*buffer*/, size_t /*size*/, void* user) {
    auto* promise = static_cast<std::promise<void>*>(user);
    promise->set_value();
  };

  filament::backend::PixelBufferDescriptor pbd(
      result.pixels.data(),
      result.pixels.size(),
      filament::backend::PixelDataFormat::RGBA,
      filament::backend::PixelDataType::UBYTE,
      callback,
      &readbackPromise);

  // Read from distortion target when distortion is active, otherwise from main target
  auto* readTarget = _distortionTarget ? _distortionTarget->GetFilamentRenderTarget()
                                       : _renderTarget->GetFilamentRenderTarget();

  _renderer->readPixels(
      readTarget,
      0,
      0,
      static_cast<uint32_t>(_config.width),
      static_cast<uint32_t>(_config.height),
      std::move(pbd));

  // Flush GPU commands to trigger the readback
  _engine->flushAndWait();

  // Wait for the readback callback
  readbackFuture.wait();

  return result;
}

std::future<RenderResult> ObservationCamera::ReadPixelsAsync() {
  // Heap-allocated state that outlives this function. Ownership is transferred
  // to the readback callback, which deletes it after fulfilling the promise.
  struct AsyncReadback {
    RenderResult result;
    std::promise<RenderResult> promise;
  };
  auto* state = new AsyncReadback();
  state->result.width = _config.width;
  state->result.height = _config.height;
  state->result.channels = 4;
  state->result.format = ImageFormat::RGBA8;
  state->result.pixels.resize(
      static_cast<size_t>(_config.width) * _config.height * state->result.channels);

  auto future = state->promise.get_future();

  auto callback = [](void* /*buffer*/, size_t /*size*/, void* user) {
    auto* s = static_cast<AsyncReadback*>(user);
    s->promise.set_value(std::move(s->result));
    delete s;
  };

  filament::backend::PixelBufferDescriptor pbd(
      state->result.pixels.data(),
      state->result.pixels.size(),
      filament::backend::PixelDataFormat::RGBA,
      filament::backend::PixelDataType::UBYTE,
      callback,
      state);

  auto* readTarget = _distortionTarget ? _distortionTarget->GetFilamentRenderTarget()
                                       : _renderTarget->GetFilamentRenderTarget();

  _renderer->readPixels(
      readTarget,
      0,
      0,
      static_cast<uint32_t>(_config.width),
      static_cast<uint32_t>(_config.height),
      std::move(pbd));

  // NOTE: The caller is responsible for flushing to submit these commands
  // to the backend. See RenderAndReadbackAsync() for the typical pattern.
  return future;
}

std::future<RenderResult> ObservationCamera::RenderAndReadbackAsync() {
  RenderFrame();
  return ReadPixelsAsync();
  // NOTE: The caller must flush or flushAndWait after calling this.
  // Filament's readPixels callback is queued for main-thread delivery and
  // only fires during beginFrame(), engine->execute(), or flushAndWait().
  // Batching multiple cameras before a single flushAndWait() is more
  // efficient than flushing per-camera.
}

RenderResult ObservationCamera::RenderAndReadback(PipelineMode mode) {
  switch (mode) {
    case PipelineMode::Synchronized: {
      // Render this frame, wait for readback, return.
      RenderFrame();
      return ReadPixels();
    }
    case PipelineMode::OneFrameDelay: {
      // Return previous frame's result (may be empty on first call).
      RenderResult returnResult = std::move(_previousFrameResult);
      // Render current frame for next call.
      RenderFrame();
      _previousFrameResult = ReadPixels();
      _hasPreviousFrame = true;
      return returnResult;
    }
    case PipelineMode::MaxPerformance: {
      // The ring buffer renders to separate render targets so the GPU can
      // work on readback for slot N while we render to slot N+1. However,
      // on Filament's Metal backend, readPixels callbacks are only delivered
      // during flushAndWait() — not during beginFrame() or flush(). So we
      // must flushAndWait after each readPixels. The ring buffer still
      // provides the benefit of separate render targets (no overwrite risk).
      auto& slot = _ringBuffer[_ringWriteIndex];

      // If this slot has a pending result from a previous cycle, collect it.
      RenderResult returnResult;
      if (slot.pending) {
        // The flushAndWait below should have already resolved this future,
        // so this wait is effectively a no-op.
        slot.future.wait();
        returnResult = std::move(slot.result);
        slot.pending = false;
      }

      // Initialize ring buffer slot render target if needed.
      if (!slot.renderTarget) {
        slot.renderTarget = OffscreenRenderTarget::Create(_engine, _config.width, _config.height);
      } else if (slot.renderTarget->NeedResize(_config.width, _config.height)) {
        slot.renderTarget->Resize(_config.width, _config.height);
      }

      // Point the view at this slot's render target and render.
      _view->setRenderTarget(slot.renderTarget->GetFilamentRenderTarget());

      // When distortion is active, re-bind the source texture to the slot's
      // color output so the distortion shader reads from the correct target.
      if (_distortionMaterialInstance) {
        filament::TextureSampler sampler(
            filament::TextureSampler::MinFilter::LINEAR,
            filament::TextureSampler::MagFilter::LINEAR);
        _distortionMaterialInstance->setParameter(
            "sourceTexture", slot.renderTarget->GetColorTexture(), sampler);
      }

      RenderFrame();

      // Start async readback into this slot.
      slot.result.width = _config.width;
      slot.result.height = _config.height;
      slot.result.channels = 4;
      slot.result.format = ImageFormat::RGBA8;
      slot.result.pixels.resize(static_cast<size_t>(_config.width) * _config.height * 4);

      slot.promise = std::promise<void>();
      slot.future = slot.promise.get_future();

      auto callback = [](void*, size_t, void* user) {
        auto* p = static_cast<std::promise<void>*>(user);
        p->set_value();
      };

      filament::backend::PixelBufferDescriptor pbd(
          slot.result.pixels.data(),
          slot.result.pixels.size(),
          filament::backend::PixelDataFormat::RGBA,
          filament::backend::PixelDataType::UBYTE,
          callback,
          &slot.promise);

      // Read from distortion target when distortion is active, otherwise from
      // the slot's render target — matching ReadPixels() behavior.
      auto* readTarget = _distortionTarget ? _distortionTarget->GetFilamentRenderTarget()
                                           : slot.renderTarget->GetFilamentRenderTarget();

      _renderer->readPixels(
          readTarget,
          0,
          0,
          static_cast<uint32_t>(_config.width),
          static_cast<uint32_t>(_config.height),
          std::move(pbd));

      // NOTE: flushAndWait() is called per-camera because Metal's readPixels callback
      // only fires during flushAndWait(), not during beginFrame(). On Vulkan, the
      // callback fires during beginFrame(), so this synchronous flush is unnecessary
      // and makes MaxPerformance mode equivalent to Synchronized mode.
      // TODO: Make this conditional on the backend to enable true async pipelining on Vulkan.
      _engine->flushAndWait();

      slot.pending = true;
      _ringWriteIndex = (_ringWriteIndex + 1) % kMaxRingBufferSize;

      // Restore primary render target and distortion source texture for
      // future Synchronized/OneFrameDelay calls.
      _view->setRenderTarget(_renderTarget->GetFilamentRenderTarget());
      if (_distortionMaterialInstance) {
        filament::TextureSampler sampler(
            filament::TextureSampler::MinFilter::LINEAR,
            filament::TextureSampler::MagFilter::LINEAR);
        _distortionMaterialInstance->setParameter(
            "sourceTexture", _renderTarget->GetColorTexture(), sampler);
      }

      return returnResult;
    }
  }
  return {};
}

void ObservationCamera::UpdateProjection() {
  if (_cameraModel.HasIntrinsics()) {
    // Build OpenGL projection matrix from OpenCV intrinsics.
    // OpenCV: x_pixel = fx * X/Z + cx, y_pixel = fy * Y/Z + cy
    // We need to map to OpenGL NDC [-1, 1] for both axes.
    //
    // The OpenGL projection matrix (column-major) maps camera-space (x,y,z)
    // to clip coordinates, which after perspective divide give NDC.
    // For a pinhole camera with intrinsics (fx, fy, cx, cy) and image size (w, h):
    //
    //   NDC_x = 2*fx/w * X/Z + (2*cx/w - 1)  maps to [-1, 1]
    //   NDC_y = 2*fy/h * Y/Z + (2*cy/h - 1)  maps to [-1, 1]  (but Y is flipped)
    //
    auto w = static_cast<float>(_config.width);
    auto h = static_cast<float>(_config.height);
    float fx = _cameraModel.fx;
    float fy = _cameraModel.fy;
    float cx = _cameraModel.cx;
    float cy = _cameraModel.cy;
    double n = _config.nearPlane;
    double f = _config.farPlane;

    // Filament uses OpenGL conventions: camera looks down -Z, Y is up.
    // OpenCV convention: camera looks down +Z, Y is down.
    // The model matrix handles the coordinate system conversion.
    // The projection matrix maps OpenGL camera space to NDC.
    //
    // In OpenGL camera space (right-handed, -Z forward):
    //   x_ndc = (2*fx/w) * (X / -Z) - (2*cx/w - 1)
    //   y_ndc = (2*fy/h) * (-Y / -Z) - (2*cy/h - 1)
    //
    // As a 4x4 projection matrix (row notation, then transposed for column-major):
    filament::math::mat4 proj(0.0);
    proj[0][0] = 2.0 * fx / w;
    proj[1][1] = 2.0 * fy / h;
    proj[2][0] = 1.0 - 2.0 * cx / w;
    proj[2][1] = 2.0 * cy / h - 1.0;
    // Standard OpenGL depth mapping for [near, far] -> [-1, 1]
    proj[2][2] = -(f + n) / (f - n);
    proj[2][3] = -1.0;
    proj[3][2] = -2.0 * f * n / (f - n);

    _camera->setCustomProjection(FlipProjectionXY(proj), proj, n, f);
  } else {
    double n = _config.nearPlane;
    double f = _config.farPlane;
    float aspect = static_cast<float>(_config.width) / static_cast<float>(_config.height);
    auto proj = filament::Camera::projection(
        filament::Camera::Fov::VERTICAL, _config.fovDegrees, aspect, n, f);
    _camera->setCustomProjection(FlipProjectionXY(proj), proj, n, f);
  }
}

void ObservationCamera::SetCameraModel(OpenCVCameraModel const& model) {
  _cameraModel = model;
  if (model.HasDistortion()) {
    SetupDistortionPass();
  } else {
    DestroyDistortionPass();
  }
  UpdateProjection();
}

void ObservationCamera::SetupDistortionPass() {
  if (_cameraModel.k4 != 0.0 || _cameraModel.k5 != 0.0 || _cameraModel.k6 != 0.0) {
    printf(
        "[mochi_renderer] WARNING: k4/k5/k6 distortion coefficients are set but NOT supported by "
        "the shader. They will be ignored.\n");
  }

  // Clean up any existing distortion pass
  DestroyDistortionPass();

  auto& em = utils::EntityManager::get();

  // Create the distortion render target (same size as main target)
  _distortionTarget = OffscreenRenderTarget::Create(_engine, _config.width, _config.height);

  // Build the distortion material from compiled resource
  _distortionMaterial =
      filament::Material::Builder()
          .package(
              MOCHI_RENDERER_MATERIALS_DISTORTION_DATA, MOCHI_RENDERER_MATERIALS_DISTORTION_SIZE)
          .build(*_engine);
  _distortionMaterialInstance = _distortionMaterial->createInstance();

  // Set distortion uniforms
  _distortionMaterialInstance->setParameter("fx", _cameraModel.fx);
  _distortionMaterialInstance->setParameter("fy", _cameraModel.fy);
  _distortionMaterialInstance->setParameter("cx", _cameraModel.cx);
  _distortionMaterialInstance->setParameter("cy", _cameraModel.cy);
  _distortionMaterialInstance->setParameter("k1", _cameraModel.k1);
  _distortionMaterialInstance->setParameter("k2", _cameraModel.k2);
  _distortionMaterialInstance->setParameter("k3", _cameraModel.k3);
  _distortionMaterialInstance->setParameter("p1", _cameraModel.p1);
  _distortionMaterialInstance->setParameter("p2", _cameraModel.p2);
  _distortionMaterialInstance->setParameter(
      "imageSize",
      filament::math::float2{
          static_cast<float>(_config.width), static_cast<float>(_config.height)});

  // Bind the undistorted render's color texture as input
  filament::TextureSampler sampler(
      filament::TextureSampler::MinFilter::LINEAR, filament::TextureSampler::MagFilter::LINEAR);
  _distortionMaterialInstance->setParameter(
      "sourceTexture", _renderTarget->GetColorTexture(), sampler);

  // Create fullscreen triangle geometry
  // Oversized triangle covers the entire viewport after clipping
  struct Vertex {
    filament::math::float2 position;
    filament::math::float2 uv;
  };
  static Vertex const kVertices[] = {
      {.position = {-1.0f, -1.0f}, .uv = {0.0f, 0.0f}},
      {.position = {3.0f, -1.0f}, .uv = {2.0f, 0.0f}},
      {.position = {-1.0f, 3.0f}, .uv = {0.0f, 2.0f}},
  };
  static uint16_t const kIndices[] = {0, 1, 2};

  _distortionVB = filament::VertexBuffer::Builder()
                      .vertexCount(3)
                      .bufferCount(1)
                      .attribute(
                          filament::VertexAttribute::POSITION,
                          0,
                          filament::backend::ElementType::FLOAT2,
                          0,
                          sizeof(Vertex))
                      .attribute(
                          filament::VertexAttribute::UV0,
                          0,
                          filament::backend::ElementType::FLOAT2,
                          sizeof(filament::math::float2),
                          sizeof(Vertex))
                      .build(*_engine);
  _distortionVB->setBufferAt(
      *_engine, 0, filament::VertexBuffer::BufferDescriptor(kVertices, sizeof(kVertices)));

  _distortionIB = filament::IndexBuffer::Builder()
                      .indexCount(3)
                      .bufferType(filament::IndexBuffer::IndexType::USHORT)
                      .build(*_engine);
  _distortionIB->setBuffer(
      *_engine, filament::IndexBuffer::BufferDescriptor(kIndices, sizeof(kIndices)));

  // Create the fullscreen triangle entity
  _distortionQuadEntity = em.create();
  filament::RenderableManager::Builder(1)
      .geometry(
          0,
          filament::RenderableManager::PrimitiveType::TRIANGLES,
          _distortionVB,
          _distortionIB,
          0,
          3)
      .material(0, _distortionMaterialInstance)
      .culling(false)
      .receiveShadows(false)
      .castShadows(false)
      .boundingBox({.center = {-1, -1, -1}, .halfExtent = {1, 1, 1}})
      .build(*_engine, _distortionQuadEntity);

  // Create a separate scene with just the fullscreen triangle
  _distortionScene = _engine->createScene();
  _distortionScene->addEntity(_distortionQuadEntity);

  // Create an orthographic camera for the distortion pass
  _distortionCameraEntity = em.create();
  _distortionCamera = _engine->createCamera(_distortionCameraEntity);
  _distortionCamera->setProjection(filament::Camera::Projection::ORTHO, -1, 1, -1, 1, 0, 1);

  // Create the distortion view
  _distortionView = _engine->createView();
  _distortionView->setScene(_distortionScene);
  _distortionView->setCamera(_distortionCamera);
  _distortionView->setRenderTarget(_distortionTarget->GetFilamentRenderTarget());
  _distortionView->setViewport(
      filament::Viewport{
          0, 0, static_cast<uint32_t>(_config.width), static_cast<uint32_t>(_config.height)});
  _distortionView->setPostProcessingEnabled(false);
  _distortionView->setShadowingEnabled(false);
}

void ObservationCamera::DestroyDistortionPass() {
  auto& em = utils::EntityManager::get();

  if (_distortionView) {
    _engine->destroy(_distortionView);
    _distortionView = nullptr;
  }
  if (_distortionCamera) {
    _engine->destroyCameraComponent(_distortionCameraEntity);
    em.destroy(_distortionCameraEntity);
    _distortionCamera = nullptr;
  }
  if (_distortionScene) {
    _engine->destroy(_distortionScene);
    _distortionScene = nullptr;
  }
  if (_distortionQuadEntity) {
    _engine->destroy(_distortionQuadEntity);
    em.destroy(_distortionQuadEntity);
    _distortionQuadEntity = {};
  }
  if (_distortionIB) {
    _engine->destroy(_distortionIB);
    _distortionIB = nullptr;
  }
  if (_distortionVB) {
    _engine->destroy(_distortionVB);
    _distortionVB = nullptr;
  }
  if (_distortionMaterialInstance) {
    _engine->destroy(_distortionMaterialInstance);
    _distortionMaterialInstance = nullptr;
  }
  if (_distortionMaterial) {
    _engine->destroy(_distortionMaterial);
    _distortionMaterial = nullptr;
  }
  _distortionTarget.reset();
}

void ObservationCamera::UpdateAttachedTransform() {
  // Find the named object in the scene and compose transforms
  auto const* sceneObjects = &_scene->GetSceneObjects();
  for (auto* obj : *sceneObjects) {
    if (obj->GetName() == _attachedObjectName) {
      auto parentTransform = obj->GetWorldTransform();
      filament::math::mat4f localModel(_localRotation);
      localModel[3] = filament::math::float4{_localPosition, 1.0f};
      auto worldModel = parentTransform * localModel;
      _camera->setModelMatrix(filament::math::mat4(worldModel));
      return;
    }
  }
  // If object not found, use local transform directly
  filament::math::mat4f model(_localRotation);
  model[3] = filament::math::float4{_localPosition, 1.0f};
  _camera->setModelMatrix(model);
}

} // namespace mochi_renderer
