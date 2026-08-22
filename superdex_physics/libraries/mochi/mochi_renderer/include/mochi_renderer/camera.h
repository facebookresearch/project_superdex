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

#include <mochi_renderer/types.h>

#include <filament/Camera.h>
#include <filament/ColorGrading.h>
#include <filament/Engine.h>
#include <filament/Material.h>
#include <filament/Renderer.h>
#include <filament/Scene.h>
#include <filament/SwapChain.h>
#include <filament/View.h>
#include <utils/Entity.h>

#include <array>
#include <future>
#include <memory>
#include <string>

#include <utils/unwindows.h> // Clean up Windows macros (near, far, OPAQUE, etc.)

namespace mochi_renderer {

class Scene;
class OffscreenRenderTarget;

// An observation camera renders to an offscreen target and supports pixel readback.
// Used for TCP/gym observation images — not displayed in a window.
class ObservationCamera {
 public:
  struct Config {
    std::string name;
    int width = 640;
    int height = 480;
    float fovDegrees = 60.0f;
    float nearPlane = 0.01f;
    float farPlane = 100.0f;
  };

  ObservationCamera(
      filament::Engine* engine,
      Scene* scene,
      filament::Renderer* renderer,
      Config const& config);
  ~ObservationCamera();

  // Non-copyable
  ObservationCamera(ObservationCamera const&) = delete;
  ObservationCamera& operator=(ObservationCamera const&) = delete;

  // --- Transform ---
  void SetTransform(filament::math::double3 position, filament::math::quat rotation);
  void LookAt(filament::math::double3 eye, filament::math::double3 target);
  filament::math::double3 GetPosition() const;
  filament::math::quat GetRotation() const;

  // --- Configuration ---
  void SetResolution(int width, int height);
  void SetFov(float degrees);
  void SetClipPlanes(float nearPlane, float farPlane);

  // Set OpenCV camera model (intrinsics + distortion).
  // When set, the projection matrix is derived from intrinsics instead of FOV,
  // and distortion is applied as a GPU post-process pass.
  void SetCameraModel(OpenCVCameraModel const& model);
  OpenCVCameraModel const& GetCameraModel() const {
    return _cameraModel;
  }

  // Apply view settings (color grading, tone mapping, post-processing) to the observation camera.
  // This ensures proper exposure and tone mapping with high-intensity lighting.
  void ApplyViewSettings(SceneViewSettings const& settings);

  // --- Attachment ---
  // Attach this camera relative to a named scene object.
  // When attached, the camera's transform is relative to the object.
  void AttachToObject(std::string const& objectName);
  void Detach();
  bool IsAttached() const {
    return !_attachedObjectName.empty();
  }
  std::string const& GetAttachedObjectName() const {
    return _attachedObjectName;
  }

  // --- Rendering ---
  // Render this camera's view to its offscreen target.
  void RenderFrame();

  // --- Readback ---
  // Synchronous pixel readback. Blocks until GPU->CPU copy completes.
  RenderResult ReadPixels();

  // Asynchronous pixel readback. Kicks off GPU readback and flushes commands,
  // but does not block. Returns a future that resolves when pixels are ready.
  std::future<RenderResult> ReadPixelsAsync();

  // Pipeline-mode-aware render + readback.
  // Returns the RenderResult appropriate for the current pipeline mode.
  RenderResult RenderAndReadback(PipelineMode mode);

  // Async render + readback. Renders the frame, kicks off readback, and returns
  // a future. The caller can do other work (scene updates, etc.) while the GPU
  // works. Call future.get() to block until the result is ready.
  std::future<RenderResult> RenderAndReadbackAsync();

  // --- Accessors ---
  Config const& GetConfig() const {
    return _config;
  }
  std::string const& GetName() const {
    return _config.name;
  }

 private:
  void UpdateProjection();
  void UpdateAttachedTransform();
  void SetupDistortionPass();
  void DestroyDistortionPass();
  void RenderDistortionPass();

  filament::Engine* _engine = nullptr;
  Scene* _scene = nullptr;
  filament::Renderer* _renderer = nullptr;
  Config _config;

  // Cached headless swap chain — created once, recreated on resolution change.
  filament::SwapChain* _swapChain = nullptr;

  utils::Entity _cameraEntity;
  filament::Camera* _camera = nullptr;
  filament::View* _view = nullptr;
  std::unique_ptr<OffscreenRenderTarget> _renderTarget;

  std::string _attachedObjectName;
  filament::math::double3 _localPosition = {0, 0, 0};
  filament::math::quat _localRotation = {0, 0, 0, 1};

  // Color grading for tone mapping — owned by this camera
  filament::ColorGrading* _colorGrading = nullptr;

  // OpenCV camera model
  OpenCVCameraModel _cameraModel;

  // GPU distortion pass resources
  std::unique_ptr<OffscreenRenderTarget> _distortionTarget;
  filament::Scene* _distortionScene = nullptr;
  filament::View* _distortionView = nullptr;
  filament::Camera* _distortionCamera = nullptr;
  utils::Entity _distortionCameraEntity;
  filament::Material* _distortionMaterial = nullptr;
  filament::MaterialInstance* _distortionMaterialInstance = nullptr;
  utils::Entity _distortionQuadEntity;
  filament::VertexBuffer* _distortionVB = nullptr;
  filament::IndexBuffer* _distortionIB = nullptr;

  // For OneFrameDelay mode
  RenderResult _previousFrameResult;
  bool _hasPreviousFrame = false;

  // For MaxPerformance mode — ring of render targets
  static constexpr int kMaxRingBufferSize = 3;
  struct InFlightFrame {
    std::unique_ptr<OffscreenRenderTarget> renderTarget;
    RenderResult result;
    std::promise<void> promise;
    std::future<void> future;
    bool pending = false;
  };
  std::array<InFlightFrame, kMaxRingBufferSize> _ringBuffer;
  int _ringWriteIndex = 0;
};

} // namespace mochi_renderer
