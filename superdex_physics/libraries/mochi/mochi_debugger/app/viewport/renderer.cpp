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

#include <mochi_renderer/windows_compat.h> // Must be first — cleans up Windows macros before Filament

#include "render_target.h"
#include "renderer.h"

#include <mochi_core/mochi_platform.h>
#include <mochi_core/utils/debug.h>
#include <mochi_renderer/scene.h>

#include <filament/Engine.h>
#include <filament/SwapChain.h>
#include <imguios/common.h>

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

void Renderer::Render(mochi_renderer::Scene* scene, RenderTarget* renderTarget, bool flushAndWait)
    const {
  if (_renderer->beginFrame(_swapChain)) {
    if (scene->_view->getRenderTarget() != renderTarget->_renderTarget) {
      scene->_view->setRenderTarget(renderTarget->_renderTarget);
    }
    _renderer->render(scene->_view);
    _renderer->endFrame();
    if (flushAndWait) {
      _engine->flushAndWait();
    }
  }
#if !MOCHI_PLATFORM_MACOS
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
#endif
}
