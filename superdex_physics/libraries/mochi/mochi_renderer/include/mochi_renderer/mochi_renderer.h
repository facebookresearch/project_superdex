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

#include <filament/Engine.h>
#include <filament/Renderer.h>
#include <filament/SwapChain.h>

#include <backend/DriverEnums.h>

#include <math/quat.h>
#include <math/vec3.h>
#include <utils/Entity.h>
#include <utils/EntityManager.h>

#include <future>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <utils/unwindows.h> // Clean up Windows macros (near, far, OPAQUE, etc.)

namespace mochi_renderer {

class Scene;
class ResourceManager;
class ObservationCamera;
class OffscreenRenderTarget;

// Top-level renderer class. Owns the Filament Engine, manages cameras, scenes,
// and quality settings. No dependency on ImGui, GLFW, or any windowing system.
// Window handles are injected optionally by host applications.
class MochiRenderer {
 public:
  struct Config {
    PipelineMode pipelineMode = PipelineMode::Synchronized;
    SceneViewSettings viewSettings = {};
    // Backend selection: VULKAN on Linux/Windows, METAL on macOS.
    // Set to DEFAULT to let Filament auto-select.
    filament::Engine::Backend backend = filament::Engine::Backend::DEFAULT;
    // OpenGL shared context for interop with host app (e.g., ImGuiOS).
    // Set to non-null to use Engine::Builder with sharedContext().
    void* sharedContext = nullptr;
    // Filament feature level (only used with sharedContext).
    filament::backend::FeatureLevel featureLevel = filament::backend::FeatureLevel::FEATURE_LEVEL_1;
    // Filament command buffer size in MiB. Default of 64 prevents
    // CircularBuffer overflow when many material instances are loaded.
    // Filament default is 3 MiB. Our usage of CreateAssetInstanced()
    // With a fixed number of 32 instances per asset was causing huge frames when we loaded many
    // meshes in the same frame. (n meshes * n instances)
    uint32_t commandBufferSizeMB = 64;
    // Filament handle allocator arena size in MiB. Default of 32 prevents
    // fallback to system heap when many GPU resources are alive.
    uint32_t driverHandleArenaSizeMB = 32;
  };

  static std::unique_ptr<MochiRenderer> Create();
  static std::unique_ptr<MochiRenderer> Create(Config const& config);
  ~MochiRenderer();

  // Non-copyable, non-movable
  MochiRenderer(MochiRenderer const&) = delete;
  MochiRenderer& operator=(MochiRenderer const&) = delete;

  // --- Engine access ---
  filament::Engine* GetEngine() const {
    return _engine;
  }
  filament::Renderer* GetFilamentRenderer() const {
    return _renderer;
  }

  // --- Scene ---
  Scene* GetScene() const {
    return _scene.get();
  }
  ResourceManager* GetResourceManager() const {
    return _resourceManager.get();
  }

  // --- Presentation (window) ---
  // Register a window for presentation rendering. The host app provides the
  // native window handle (HWND, NSWindow*, ANativeWindow*, etc.).
  // Only one presentation target is supported.
  void SetPresentationTarget(void* nativeWindowHandle, int width, int height);
  void ResizePresentationTarget(int width, int height);
  bool HasPresentationTarget() const {
    return _presentationSwapChain != nullptr;
  }

  // --- Observation cameras ---
  // Create a named observation camera that renders to an offscreen target.
  // Returns nullptr if a camera with that name already exists.
  ObservationCamera* CreateObservationCamera(std::string const& name, int width, int height);
  ObservationCamera* GetObservationCamera(std::string const& name) const;
  std::vector<std::string> GetObservationCameraNames() const;
  void DestroyObservationCamera(std::string const& name);

  // --- Rendering ---
  // Render a single frame.
  // - Renders the presentation camera to the window swap chain (if set).
  // - Renders requested observation cameras to their offscreen targets.
  // - readbackCameras: names of cameras to read back pixels for.
  //   If empty and in Synchronized mode, no observation cameras render.
  void Render(std::vector<std::string> const& readbackCameras = {});

  // Render requested cameras and return results keyed by camera name.
  std::unordered_map<std::string, RenderResult> RenderAndReadback(
      std::vector<std::string> const& cameraNames);

  // Async variant: renders cameras and returns futures that resolve when pixels
  // are ready. The caller can do other work while the GPU renders and reads back.
  // Use this for pipelining — overlap scene updates with GPU rendering.
  std::unordered_map<std::string, std::future<RenderResult>> RenderAndReadbackAsync(
      std::vector<std::string> const& cameraNames);

  // --- Pipeline mode ---
  void SetPipelineMode(PipelineMode mode);
  PipelineMode GetPipelineMode() const {
    return _pipelineMode;
  }

  // --- Quality settings ---
  void SetViewSettings(SceneViewSettings const& settings);
  SceneViewSettings const& GetViewSettings() const {
    return _viewSettings;
  }

  // --- Presentation camera control ---
  // These control the camera used for window presentation.
  void SetPresentationCameraTransform(
      filament::math::double3 position,
      filament::math::quat rotation);
  void SetPresentationCameraLookAt(
      filament::math::double3 eye,
      filament::math::double3 target,
      filament::math::double3 up = {0, 0, 1});
  void SetPresentationCameraProjection(float fovDegrees, float nearPlane, float farPlane);
  void SetPresentationViewport(int width, int height);

 private:
  MochiRenderer() = default;

  filament::Engine* _engine = nullptr;
  filament::Renderer* _renderer = nullptr;
  filament::SwapChain* _presentationSwapChain = nullptr;

  std::unique_ptr<Scene> _scene;
  std::unique_ptr<ResourceManager> _resourceManager;

  std::unordered_map<std::string, std::unique_ptr<ObservationCamera>> _observationCameras;

  PipelineMode _pipelineMode = PipelineMode::Synchronized;
  SceneViewSettings _viewSettings;

  // For presentation camera
  utils::Entity _presentationCameraEntity;
  filament::Camera* _presentationCamera = nullptr;
  filament::View* _presentationView = nullptr;
};

} // namespace mochi_renderer
