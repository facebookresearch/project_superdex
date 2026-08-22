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

#include <filament/Engine.h>
#include <filament/RenderTarget.h>
#include <filament/Texture.h>

#include <memory>

namespace mochi_renderer {

// Filament-native offscreen render target. No OpenGL dependency.
// Renders to GPU textures that can be read back via filament::Renderer::readPixels().
class OffscreenRenderTarget {
 public:
  static std::unique_ptr<OffscreenRenderTarget>
  Create(filament::Engine* engine, int width, int height);
  ~OffscreenRenderTarget();

  bool NeedResize(int newWidth, int newHeight) const;
  void Resize(int width, int height);
  void GetSize(int& width, int& height) const;

  filament::RenderTarget* GetFilamentRenderTarget() const {
    return _renderTarget;
  }
  filament::Texture* GetColorTexture() const {
    return _colorTexture;
  }

 private:
  OffscreenRenderTarget() = default;
  void DestroyResources();

  filament::Engine* _engine = nullptr;
  filament::Texture* _colorTexture = nullptr;
  filament::Texture* _depthTexture = nullptr;
  filament::RenderTarget* _renderTarget = nullptr;
  int _width = 0;
  int _height = 0;
};

} // namespace mochi_renderer
