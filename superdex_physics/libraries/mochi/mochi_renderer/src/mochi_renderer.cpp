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
#include <mochi_renderer/mochi_renderer.h>
#include <mochi_renderer/resource_manager.h>
#include <mochi_renderer/scene.h>

#include <filament/Camera.h>
#include <filament/View.h>
#include <filament/Viewport.h>
#include <utils/EntityManager.h>

#include <mochi_core/utils/debug.h>

namespace mochi_renderer {

std::unique_ptr<MochiRenderer> MochiRenderer::Create() {
  Config config;
  return Create(config);
}

std::unique_ptr<MochiRenderer> MochiRenderer::Create(Config const& config) {
  auto renderer = std::unique_ptr<MochiRenderer>(new MochiRenderer());

  // Select backend
  auto backend = config.backend;
  if (backend == filament::Engine::Backend::DEFAULT) {
#if defined(__APPLE__)
    backend = filament::Engine::Backend::METAL;
#else
    backend = filament::Engine::Backend::VULKAN;
#endif
  }

  filament::Engine::Config engineConfig;
  engineConfig.commandBufferSizeMB = config.commandBufferSizeMB;
  engineConfig.driverHandleArenaSizeMB = config.driverHandleArenaSizeMB;

  if (config.sharedContext) {
    renderer->_engine = filament::Engine::Builder()
                            .backend(backend)
                            .sharedContext(config.sharedContext)
                            .featureLevel(config.featureLevel)
                            .config(&engineConfig)
                            .build();
  } else {
    renderer->_engine = filament::Engine::create(backend, nullptr, nullptr, &engineConfig);
  }
  if (renderer->_engine == nullptr) {
    return nullptr;
  }

  renderer->_renderer = renderer->_engine->createRenderer();
  MOCHI_ASSERT(renderer->_renderer != nullptr);

  // Set clear color
  filament::Renderer::ClearOptions clearOptions;
  clearOptions.clearColor = {0.0f, 0.0f, 0.0f, 1.0f};
  clearOptions.clear = true;
  renderer->_renderer->setClearOptions(clearOptions);

  // Create scene and resource manager
  renderer->_viewSettings = config.viewSettings;
  renderer->_scene = Scene::Create(renderer->_engine, config.viewSettings);
  renderer->_resourceManager = ResourceManager::Create(renderer->_engine);

  renderer->_pipelineMode = config.pipelineMode;

  return renderer;
}

MochiRenderer::~MochiRenderer() {
  if (!_engine) {
    return;
  }

  // Destroy observation cameras first (they reference the engine)
  _observationCameras.clear();

  // Destroy presentation view/camera
  if (_presentationView) {
    _engine->destroy(_presentationView);
  }
  if (_presentationCamera) {
    _engine->destroyCameraComponent(_presentationCameraEntity);
    utils::EntityManager::get().destroy(_presentationCameraEntity);
  }
  if (_presentationSwapChain) {
    _engine->destroy(_presentationSwapChain);
  }

  // Destroy scene and resource manager before engine
  _scene.reset();
  _resourceManager.reset();

  // Destroy renderer and engine last
  _engine->destroy(_renderer);
  filament::Engine::destroy(&_engine);
}

void MochiRenderer::SetPresentationTarget(void* nativeWindowHandle, int width, int height) {
  MOCHI_ASSERT(nativeWindowHandle != nullptr);

  // Destroy existing presentation resources
  if (_presentationSwapChain) {
    _engine->destroy(_presentationSwapChain);
    _presentationSwapChain = nullptr;
  }
  if (_presentationView) {
    _engine->destroy(_presentationView);
    _presentationView = nullptr;
  }
  if (_presentationCamera) {
    _engine->destroyCameraComponent(_presentationCameraEntity);
    utils::EntityManager::get().destroy(_presentationCameraEntity);
    _presentationCamera = nullptr;
  }

  // Create swap chain from native window
  _presentationSwapChain = _engine->createSwapChain(nativeWindowHandle);

  // Create presentation camera and view
  _presentationCameraEntity = utils::EntityManager::get().create();
  _presentationCamera = _engine->createCamera(_presentationCameraEntity);
  _presentationCamera->lookAt({1.0, 1.0, 0.5}, {0.0, 0.0, 0.0}, {0, 0, 1});

  _presentationView = _engine->createView();
  _presentationView->setScene(_scene->_scene);
  _presentationView->setCamera(_presentationCamera);
  _presentationView->setViewport(
      filament::Viewport{0, 0, static_cast<uint32_t>(width), static_cast<uint32_t>(height)});

  float aspect = static_cast<float>(width) / static_cast<float>(height);
  _presentationCamera->setProjection(60.0f, aspect, 0.01f, 100.0f);

  _scene->ApplyViewSettings(_viewSettings);
}

void MochiRenderer::ResizePresentationTarget(int width, int height) {
  if (!_presentationView || !_presentationCamera) {
    return;
  }
  _presentationView->setViewport(
      filament::Viewport{0, 0, static_cast<uint32_t>(width), static_cast<uint32_t>(height)});
  float aspect = static_cast<float>(width) / static_cast<float>(height);
  _presentationCamera->setProjection(
      _presentationCamera->getFieldOfViewInDegrees(filament::Camera::Fov::VERTICAL),
      aspect,
      _presentationCamera->getNear(),
      _presentationCamera->getCullingFar());
}

ObservationCamera*
MochiRenderer::CreateObservationCamera(std::string const& name, int width, int height) {
  if (_observationCameras.contains(name)) {
    return nullptr;
  }
  ObservationCamera::Config config;
  config.name = name;
  config.width = width;
  config.height = height;
  auto camera = std::make_unique<ObservationCamera>(_engine, _scene.get(), _renderer, config);

  // Apply current view settings (tone mapping, post-processing) to the new camera
  camera->ApplyViewSettings(_viewSettings);

  auto* raw = camera.get();
  _observationCameras[name] = std::move(camera);
  return raw;
}

ObservationCamera* MochiRenderer::GetObservationCamera(std::string const& name) const {
  auto it = _observationCameras.find(name);
  if (it != _observationCameras.end()) {
    return it->second.get();
  }
  return nullptr;
}

std::vector<std::string> MochiRenderer::GetObservationCameraNames() const {
  std::vector<std::string> names;
  names.reserve(_observationCameras.size());
  for (auto const& [name, _] : _observationCameras) {
    names.push_back(name);
  }
  return names;
}

void MochiRenderer::DestroyObservationCamera(std::string const& name) {
  _observationCameras.erase(name);
}

void MochiRenderer::Render(std::vector<std::string> const& readbackCameras) {
  // Render to presentation swap chain if available
  if (_presentationSwapChain && _presentationView) {
    if (_renderer->beginFrame(_presentationSwapChain)) {
      _renderer->render(_presentationView);
      _renderer->endFrame();
    }
  }

  // Render observation cameras that were requested
  for (auto const& cameraName : readbackCameras) {
    auto it = _observationCameras.find(cameraName);
    if (it != _observationCameras.end()) {
      it->second->RenderFrame();
    }
  }
}

std::unordered_map<std::string, RenderResult> MochiRenderer::RenderAndReadback(
    std::vector<std::string> const& cameraNames) {
  // Render presentation first
  if (_presentationSwapChain && _presentationView) {
    if (_renderer->beginFrame(_presentationSwapChain)) {
      _renderer->render(_presentationView);
      _renderer->endFrame();
    }
  }

  // Render each requested observation camera
  std::unordered_map<std::string, RenderResult> results;
  for (auto const& name : cameraNames) {
    auto it = _observationCameras.find(name);
    if (it != _observationCameras.end()) {
      results[name] = it->second->RenderAndReadback(_pipelineMode);
    }
  }
  return results;
}

std::unordered_map<std::string, std::future<RenderResult>> MochiRenderer::RenderAndReadbackAsync(
    std::vector<std::string> const& cameraNames) {
  // Render presentation first
  if (_presentationSwapChain && _presentationView) {
    if (_renderer->beginFrame(_presentationSwapChain)) {
      _renderer->render(_presentationView);
      _renderer->endFrame();
    }
  }

  // Render ALL cameras and enqueue ALL readPixels commands before flushing.
  // This batches GPU work so we only synchronize once at the end.
  std::unordered_map<std::string, std::future<RenderResult>> futures;
  for (auto const& name : cameraNames) {
    auto it = _observationCameras.find(name);
    if (it != _observationCameras.end()) {
      futures[name] = it->second->RenderAndReadbackAsync();
    }
  }

  // Single flushAndWait drains all readPixels callbacks at once.
  // Filament delivers readPixels callbacks on the calling thread during
  // flushAndWait/beginFrame/execute — without this, the futures never resolve.
  if (!futures.empty()) {
    _engine->flushAndWait();
  }

  return futures;
}

void MochiRenderer::SetPipelineMode(PipelineMode mode) {
  _pipelineMode = mode;
}

void MochiRenderer::SetViewSettings(SceneViewSettings const& settings) {
  _viewSettings = settings;
  _scene->ApplyViewSettings(settings);

  // Propagate settings to all observation cameras
  for (auto& [name, camera] : _observationCameras) {
    camera->ApplyViewSettings(settings);
  }
}

void MochiRenderer::SetPresentationCameraTransform(
    filament::math::double3 position,
    filament::math::quat rotation) {
  if (_presentationCamera) {
    filament::math::mat4f model(rotation);
    model[3] = filament::math::float4{position, 1.0f};
    _presentationCamera->setModelMatrix(model);
  }
}

void MochiRenderer::SetPresentationCameraLookAt(
    filament::math::double3 eye,
    filament::math::double3 target,
    filament::math::double3 up) {
  if (_presentationCamera) {
    _presentationCamera->lookAt(eye, target, up);
  }
}

void MochiRenderer::SetPresentationCameraProjection(
    float fovDegrees,
    float nearPlane,
    float farPlane) {
  if (_presentationCamera && _presentationView) {
    auto vp = _presentationView->getViewport();
    float aspect = static_cast<float>(vp.width) / static_cast<float>(vp.height);
    _presentationCamera->setProjection(fovDegrees, aspect, nearPlane, farPlane);
  }
}

void MochiRenderer::SetPresentationViewport(int width, int height) {
  ResizePresentationTarget(width, height);
}

} // namespace mochi_renderer
