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

#include <mochi_core/mochi_platform.h>

#include <filament/Engine.h>
#include <filament/Renderer.h>

#include <memory>

namespace mochi_renderer {
class Scene;
} // namespace mochi_renderer
class RenderTarget;

// App-local Filament renderer: owns a swapchain and renders a mochi_renderer Scene into an
// offscreen RenderTarget. Lives in the global namespace so it matches the `friend class Renderer;`
// declaration in mochi_renderer/scene.h (which grants access to Scene::_view).
class Renderer {
  MOCHI_DECLARE_MOVE_ONLY(Renderer);

 public:
  static std::unique_ptr<Renderer>
  Create(filament::Engine* engine, filament::Renderer* filamentRenderer, int w, int h);
  ~Renderer();
  void SetClearColor(filament::math::float4 color);
  filament::math::float4 GetClearColor() const;
  void Render(mochi_renderer::Scene* scene, RenderTarget* renderTarget, bool flushAndWait) const;

 private:
  Renderer(filament::Engine* engine, filament::Renderer* filamentRenderer, int w, int h);

 private:
  filament::Engine* _engine = nullptr;
  filament::SwapChain* _swapChain = nullptr;
  filament::Renderer* _renderer = nullptr;
  bool _ownsRenderer = false;
};
