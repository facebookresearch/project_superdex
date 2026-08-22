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

#include <mochi_core/mochi_platform.h>
#include <mochi_core/utils/debug.h>

#include <filament/Engine.h>

#if MOCHI_PLATFORM_MACOS
// Metal backend - forward declarations (extern "C" to match render_target_metal.mm)
// The actual Metal texture creation is in render_target_metal.mm
extern "C" {
void* CreateMetalTexture(void* device, int width, int height);
void* CreateMetalTextureRefForImGui(void* filamentOwnedTexture);
void DestroyMetalTexture(void* texture);
void* GetMetalDevice();
}
#else
// OpenGL backend
#include <glad/glad.h>
#endif

std::unique_ptr<RenderTarget>
RenderTarget::Create(filament::Engine* engine, int width, int height) {
  MOCHI_ASSERT(engine != nullptr);
  auto rt = std::make_unique<RenderTarget>();
  rt->_engine = engine;
  rt->Resize(width, height);
  return rt;
}

RenderTarget::~RenderTarget() {
  DestroyResources();
}

void RenderTarget::Resize(int w, int h) {
  if (w == _width && h == _height) {
    return;
  }
  DestroyResources();
  _width = w;
  _height = h;

#if MOCHI_PLATFORM_MACOS
  // Metal backend: Create Metal texture externally and import into Filament
  void* device = GetMetalDevice();
  if (device) {
    _mtlTexture = CreateMetalTexture(device, _width, _height);
    if (_mtlTexture) {
      // Create ImGui reference BEFORE Filament import (which takes ownership)
      _imguiTextureId = reinterpret_cast<ImTextureID>(CreateMetalTextureRefForImGui(_mtlTexture));

      _colorTexture =
          filament::Texture::Builder()
              .width(_width)
              .height(_height)
              .levels(1)
              .usage(
                  filament::Texture::Usage::COLOR_ATTACHMENT | filament::Texture::Usage::SAMPLEABLE)
              .format(filament::Texture::InternalFormat::RGBA8)
              .import(reinterpret_cast<intptr_t>(_mtlTexture))
              .build(*_engine);
    }
  }
#else
  // OpenGL backend: Create GL texture and import into Filament
  glGenTextures(1, &_glTex);
  glBindTexture(GL_TEXTURE_2D, _glTex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, _width, _height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
  glBindTexture(GL_TEXTURE_2D, 0);

  // Flush so the texture is visible to Filament's shared GL context
  glFlush();

  // Store texture ID for ImGui once, avoiding repeated casts
  _imguiTextureId = reinterpret_cast<ImTextureID>(static_cast<uintptr_t>(_glTex));

  _colorTexture =
      filament::Texture::Builder()
          .width(_width)
          .height(_height)
          .levels(1)
          .usage(filament::Texture::Usage::COLOR_ATTACHMENT | filament::Texture::Usage::SAMPLEABLE)
          .format(filament::Texture::InternalFormat::RGBA8)
          .import(_glTex)
          .build(*_engine);
#endif

  _depthTexture = filament::Texture::Builder()
                      .width(_width)
                      .height(_height)
                      .levels(1)
                      .usage(filament::Texture::Usage::DEPTH_ATTACHMENT)
                      .format(filament::Texture::InternalFormat::DEPTH32F)
                      .build(*_engine);

  _renderTarget = filament::RenderTarget::Builder()
                      .texture(filament::RenderTarget::AttachmentPoint::COLOR, _colorTexture)
                      .texture(filament::RenderTarget::AttachmentPoint::DEPTH, _depthTexture)
                      .build(*_engine);
}

void RenderTarget::GetSize(int& w, int& h) const {
  w = _width;
  h = _height;
}

ImTextureID RenderTarget::GetTextureId() const {
  return _imguiTextureId;
}

void RenderTarget::DestroyResources() {
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
#if MOCHI_PLATFORM_MACOS
  // Release the ImGui Metal texture reference
  if (_imguiTextureId) {
    DestroyMetalTexture(_imguiTextureId);
  }
  _mtlTexture = nullptr;
#else
  glDeleteTextures(1, &_glTex);
  _glTex = 0;
#endif
  _imguiTextureId = nullptr;
  _width = 0;
  _height = 0;
}
