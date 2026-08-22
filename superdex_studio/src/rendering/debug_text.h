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

#include <imgui.h>

#include <math/mat4.h>
#include <math/vec3.h>
#include <math/vec4.h>

#include <string>
#include <string_view>
#include <vector>

namespace superdex::studio {

// World-anchored debug labels, mirroring the @ref mochi_renderer::DebugDraw contract:
// immediate-mode, submitted every frame, cleared by the owner. Submit from OnRender right next to
// the debug geometry the label annotates.
//
// The labels are ImGui overlay text: always on top of the viewport image, never depth-occluded.
//
// Timing: the studio runs its ImGui pass before its render pass, and the owning @ref Viewport draws
// and clears these labels at the end of the ImGui pass (@ref DebugDraw is instead cleared at the
// end of the render pass). A label submitted from OnRender therefore appears one frame later. The
// lag is imperceptible and keeps every debug primitive on a single submission site.
class DebugText {
 public:
  static constexpr filament::math::float4 kDefaultColor{1.0f, 1.0f, 1.0f, 1.0f};

  // Queues @p text centered on @p worldPosition, in render (Filament) space -- the space every
  // @ref mochi_renderer::DebugDraw entry point takes, so callers convert once and submit geometry
  // and labels with the same coordinates. @p pixelOffset nudges the label in screen space so it can
  // sit beside, rather than on top of, a marker.
  void Draw(
      filament::math::float3 worldPosition,
      std::string_view text,
      filament::math::float4 color = kDefaultColor,
      ImVec2 pixelOffset = {});

  // Projects every queued label with @p viewMatrix / @p projMatrix and emits it into the current
  // ImGui window's draw list. @p contentOrigin and @p logicalWidth / @p logicalHeight describe the
  // viewport image in ImGui (logical) coordinates. Labels behind the camera are dropped.
  void Show(
      filament::math::mat4 const& viewMatrix,
      filament::math::mat4 const& projMatrix,
      ImVec2 contentOrigin,
      float logicalWidth,
      float logicalHeight) const;

  void Clear();

 private:
  struct Item {
    filament::math::float3 worldPosition;
    std::string text;
    filament::math::float4 color;
    ImVec2 pixelOffset;
  };
  std::vector<Item> _items;
};

} // namespace superdex::studio
