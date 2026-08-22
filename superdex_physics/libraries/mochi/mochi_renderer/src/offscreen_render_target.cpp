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

#include "offscreen_render_target.h"

#include <mochi_core/utils/debug.h>

namespace mochi_renderer {

std::unique_ptr<OffscreenRenderTarget>
OffscreenRenderTarget::Create(filament::Engine* engine, int width, int height) {
  MOCHI_ASSERT(engine != nullptr);
  MOCHI_ASSERT(width > 0 && height > 0);
  auto rt = std::unique_ptr<OffscreenRenderTarget>(new OffscreenRenderTarget());
  rt->_engine = engine;
  rt->Resize(width, height);
  return rt;
}

OffscreenRenderTarget::~OffscreenRenderTarget() {
  DestroyResources();
}

bool OffscreenRenderTarget::NeedResize(int newWidth, int newHeight) const {
  return newWidth != _width || newHeight != _height;
}

void OffscreenRenderTarget::Resize(int width, int height) {
  if (width == _width && height == _height) {
    return;
  }
  DestroyResources();
  _width = width;
  _height = height;

  // Create color texture — Filament-native, no GL import.
  // BLIT_SRC is required for Renderer::readPixels() on the Vulkan backend
  // (allows vkCmdCopyImageToBuffer from this texture to a CPU buffer).
  _colorTexture = filament::Texture::Builder()
                      .width(static_cast<uint32_t>(_width))
                      .height(static_cast<uint32_t>(_height))
                      .levels(1)
                      .usage(
                          filament::Texture::Usage::COLOR_ATTACHMENT |
                          filament::Texture::Usage::SAMPLEABLE | filament::Texture::Usage::BLIT_SRC)
                      .format(filament::Texture::InternalFormat::RGBA8)
                      .build(*_engine);

  // Create depth texture
  _depthTexture = filament::Texture::Builder()
                      .width(static_cast<uint32_t>(_width))
                      .height(static_cast<uint32_t>(_height))
                      .levels(1)
                      .usage(filament::Texture::Usage::DEPTH_ATTACHMENT)
                      .format(filament::Texture::InternalFormat::DEPTH32F)
                      .build(*_engine);

  // Create render target
  _renderTarget = filament::RenderTarget::Builder()
                      .texture(filament::RenderTarget::AttachmentPoint::COLOR, _colorTexture)
                      .texture(filament::RenderTarget::AttachmentPoint::DEPTH, _depthTexture)
                      .build(*_engine);
}

void OffscreenRenderTarget::GetSize(int& width, int& height) const {
  width = _width;
  height = _height;
}

void OffscreenRenderTarget::DestroyResources() {
  if (_renderTarget) {
    _engine->destroy(_renderTarget);
    _renderTarget = nullptr;
  }
  if (_colorTexture) {
    _engine->destroy(_colorTexture);
    _colorTexture = nullptr;
  }
  if (_depthTexture) {
    _engine->destroy(_depthTexture);
    _depthTexture = nullptr;
  }
  _width = 0;
  _height = 0;
}

} // namespace mochi_renderer
