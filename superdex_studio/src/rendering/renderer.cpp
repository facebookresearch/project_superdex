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

#include "rendering/renderer.h"
#include "rendering/render_target.h"

#include <mochi_renderer/scene.h>

#include "filament/Engine.h"
#include "filament/View.h"
#include "mochi_core/mochi_platform.h"
#include "mochi_core/utils/debug.h"

#include <imguios/common.h>

#include "filament/SwapChain.h"

#include <backend/PixelBufferDescriptor.h>

#include <future>

namespace superdex::studio {

std::unique_ptr<Renderer>
Renderer::Create(filament::Engine* engine, filament::Renderer* filamentRenderer, int w, int h) {
  MOCHI_ASSERT(engine != nullptr);
  return std::unique_ptr<Renderer>(new Renderer(engine, filamentRenderer, w, h));
}

Renderer::Renderer(filament::Engine* engine, filament::Renderer* filamentRenderer, int w, int h) {
  _engine = engine;
  _swapChain = engine->createSwapChain(w, h, filament::SwapChain::CONFIG_TRANSPARENT);
  if (filamentRenderer) {
    _renderer = filamentRenderer;
    _ownsRenderer = false;
  } else {
    _renderer = engine->createRenderer();
    _ownsRenderer = true;
  }
  SetClearColor({0.0f, 0.0f, 0.0f, 1.0f});
}

Renderer::~Renderer() {
  if (_ownsRenderer) {
    _engine->destroy(_renderer);
  }
  _engine->destroy(_swapChain);
}

void Renderer::SetClearColor(filament::math::float4 color) {
  filament::Renderer::ClearOptions options;
  options.clearColor = color;
  options.clear = true;
  _renderer->setClearOptions(options);
}

filament::math::float4 Renderer::GetClearColor() const {
  return _renderer->getClearOptions().clearColor;
}

void Renderer::Render(
    mochi_renderer::Scene* scene,
    RenderTarget* renderTarget,
    bool flushAndWait,
    HighlightPass const* highlight,
    std::optional<filament::math::float4> clearColor,
    bool beginEndFrame) const {
  // Save and optionally override clear color for this render pass
  filament::Renderer::ClearOptions const savedClearOptions = _renderer->getClearOptions();
  if (clearColor) {
    filament::Renderer::ClearOptions overrideOptions = savedClearOptions;
    overrideOptions.clearColor = *clearColor;
    overrideOptions.clear = true;
    _renderer->setClearOptions(overrideOptions);
  }

  // When an explicit frame is active (BeginFrame/EndFrame), share it rather than opening a
  // nested one, so several offscreen renders present and pace only once per app frame.
  bool const ownFrame = beginEndFrame && !_frameActive;

  // Shared view rendering logic used by both frame-managed and externally-framed paths.
  auto renderViews = [&]() {
    if (scene->GetView()->getRenderTarget() != renderTarget->_renderTarget) {
      scene->GetView()->setRenderTarget(renderTarget->_renderTarget);
    }
    _renderer->render(scene->GetView());

    // Selection highlight: an isolated overlay render composited back over the main image. See
    // HighlightPass. Skipped when nothing is highlighted.
    if (highlight && highlight->overlayView && highlight->overlayTarget) {
      // Render the highlighted meshes into their own target, cleared to fully transparent so only
      // the meshes contribute coverage. Its own depth buffer (cleared via the view's channel depth
      // clear) means only the nearest surface shows and scene geometry can't occlude it.
      filament::Renderer::ClearOptions const savedClear = _renderer->getClearOptions();
      filament::Renderer::ClearOptions overlayClear = savedClear;
      overlayClear.clearColor = {0.0f, 0.0f, 0.0f, 0.0f};
      overlayClear.clear = true;
      _renderer->setClearOptions(overlayClear);
      highlight->overlayView->setRenderTarget(highlight->overlayTarget->_renderTarget);
      _renderer->render(highlight->overlayView);
      // Restore the main clear color for the next frame.
      _renderer->setClearOptions(savedClear);
    }
    // Composite pass (a TRANSLUCENT view) blends the overlay over the main target without clearing
    // it.
    if (highlight && highlight->compositeView) {
      highlight->compositeView->setRenderTarget(renderTarget->_renderTarget);
      _renderer->render(highlight->compositeView);
    }
  };

  if (ownFrame) {
    // Normal path: owns frame lifetime, respects filament frame pacing / resource gating.
    if (_renderer->beginFrame(_swapChain)) {
      renderViews();
      _renderer->endFrame();
    }
  } else {
    // Low-level path: caller guarantees a filament frame is already active on this renderer
    // (e.g., beginFrame was called externally and returned true), OR this is true offscreen
    // work with no active frame. Filament requires render() to be called between
    // beginFrame() and endFrame() - in debug builds it will invariant-check mSwapChain and
    // fail if no frame is active. For true offscreen work with *no* active frame, you can also
    // use filament::Renderer::renderStandaloneView() which must be called outside
    // beginFrame/endFrame and requires each View to have a RenderTarget.
    // Inside an explicit BeginFrame()/EndFrame() bracket, respect pacing: if beginFrame()
    // was skipped, bracketed Render() calls become no-ops.
    bool const canRender = _frameActive ? _frameStarted : true;
    if (canRender) {
      MOCHI_ASSERT(scene != nullptr);
      MOCHI_ASSERT(renderTarget != nullptr);
      renderViews();
    }
  }

  // Restore original clear color if we overrode it for this render pass
  if (clearColor) {
    _renderer->setClearOptions(savedClearOptions);
  }

  // Drain the backend when requested, even if beginFrame() skipped this frame (frame pacing).
  // Otherwise queued backend work (including pending resource destruction) is left un-flushed,
  // widening the window for use-after-free of freed/recycled handles.
  if (flushAndWait) {
    _engine->flushAndWait();
  }
#if !MOCHI_PLATFORM_MACOS
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
#endif
}

bool Renderer::BeginFrame() {
  _frameActive = true;
  _frameStarted = _renderer->beginFrame(_swapChain);
  return _frameStarted;
}

void Renderer::EndFrame() {
  if (_frameStarted) {
    _renderer->endFrame();
  }
  _frameActive = false;
  _frameStarted = false;
}

void Renderer::ReadPixels(
    RenderTarget const& target,
    std::vector<uint8_t>& outRgba,
    mochi::Error& error) const {
  MOCHI_ERROR_RETURN(error);

  int width = 0;
  int height = 0;
  target.GetSize(width, height);
  MOCHI_ERROR_IF(
      width <= 0 || height <= 0 || target._renderTarget == nullptr,
      error,
      "Cannot read pixels from an empty render target.");
  MOCHI_ERROR_RETURN(error);

  outRgba.resize(static_cast<size_t>(width) * height * 4);

  // Block until the backend fills the buffer: the callback fires once the readback completes.
  std::promise<void> readbackPromise;
  std::future<void> readbackFuture = readbackPromise.get_future();
  auto callback = [](void* /*buffer*/, size_t /*size*/, void* user) {
    static_cast<std::promise<void>*>(user)->set_value();
  };

  filament::backend::PixelBufferDescriptor pbd(
      outRgba.data(),
      outRgba.size(),
      filament::backend::PixelDataFormat::RGBA,
      filament::backend::PixelDataType::UBYTE,
      callback,
      &readbackPromise);

  _renderer->readPixels(
      target._renderTarget,
      0,
      0,
      static_cast<uint32_t>(width),
      static_cast<uint32_t>(height),
      std::move(pbd));

  // Flush GPU commands to trigger the readback, then wait for the completion callback.
  _engine->flushAndWait();
  readbackFuture.wait();
}

} // namespace superdex::studio
