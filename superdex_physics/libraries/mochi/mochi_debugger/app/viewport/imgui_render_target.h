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

#include <imgui.h>

// Helper for drawing an offscreen render-target texture in an ImGui window. The vertical flip is
// handled here via platform-specific UVs: OpenGL textures originate bottom-left (flip Y), Metal
// textures originate top-left (no flip). This keeps the UI code (gui.cpp) free of any Filament or
// windowing-backend dependency -- it only needs the texture's ImTextureID.

namespace ImGui {

#if MOCHI_PLATFORM_MACOS
// Metal: texture origin is top-left, matches ImGui - no flip needed.
inline constexpr ImVec2 kRenderTargetUV0{0, 0};
inline constexpr ImVec2 kRenderTargetUV1{1, 1};
#else
// OpenGL: texture origin is bottom-left, ImGui expects top-left - flip Y.
inline constexpr ImVec2 kRenderTargetUV0{0, 1};
inline constexpr ImVec2 kRenderTargetUV1{1, 0};
#endif

inline void RenderTargetImage(ImTextureID textureId, int width, int height) {
  float const alpha = ImGui::GetStyle().Alpha;
  ImGui::Image(
      textureId,
      ImVec2(static_cast<float>(width), static_cast<float>(height)),
      kRenderTargetUV0,
      kRenderTargetUV1,
      ImVec4(1, 1, 1, alpha));
}

} // namespace ImGui
