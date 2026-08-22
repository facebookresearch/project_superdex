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
#include "filament/RenderTarget.h"
#include "filament/Texture.h"
#include "imgui.h"

#include <mochi_core/mochi_platform.h>

namespace superdex::studio {

class Renderer;

class RenderTarget {
 public:
  static std::unique_ptr<RenderTarget> Create(filament::Engine* engine, int width, int height);
  ~RenderTarget();
  void Resize(int w, int h);
  void GetSize(int& w, int& h) const;
  ImTextureID GetTextureId() const;
  // The Filament color attachment (RGBA8, SAMPLEABLE), for sampling this target in another pass
  // (e.g. the highlight composite reads the overlay target's color).
  filament::Texture* GetColorTexture() const;

 private:
  void DestroyResources();
  friend class Renderer;
  filament::Engine* _engine = nullptr;
  filament::Texture* _colorTexture = nullptr;
  filament::Texture* _depthTexture = nullptr;
  filament::RenderTarget* _renderTarget = nullptr;
  ImTextureID _imguiTextureId = nullptr;
#if MOCHI_PLATFORM_MACOS
  void* _mtlTexture = nullptr; // Owned by Filament after import
#else
  unsigned int _tex_gl = 0;
#endif
  int _width = 0;
  int _height = 0;
};

} // namespace superdex::studio
