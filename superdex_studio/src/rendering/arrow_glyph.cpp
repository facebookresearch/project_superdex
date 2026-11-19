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

#include "rendering/arrow_glyph.h"

#include <mochi_renderer/debug.h>

#include <algorithm>

namespace superdex::studio {

void DrawArrow(
    mochi_renderer::DebugDraw& debugDraw,
    filament::math::float3 tail,
    filament::math::float3 tip,
    float shaftRadius,
    filament::math::float4 color,
    bool overlay) {
  filament::math::float3 const axis = tip - tail;
  float const axisLength = length(axis);
  if (axisLength < 1e-7f) {
    return;
  }
  filament::math::float3 const unit = axis / axisLength;
  float const shrink = std::min(1.0f, 0.5f * axisLength / (kArrowHeadLength * shaftRadius));
  float const radius = shaftRadius * shrink;
  float const headLength = kArrowHeadLength * radius;
  debugDraw.DrawSolidCylinder(
      tail, unit * (axisLength - headLength), radius, color, 12, true, overlay);
  debugDraw.DrawSolidCone(
      tail + unit * (axisLength - headLength),
      unit * headLength,
      kArrowHeadRadius * radius,
      color,
      16,
      true,
      overlay);
}

void DrawFrameAxisGlyphs(
    mochi_renderer::DebugDraw& debugDraw,
    mochi::Span<FrameAxisGlyph const> glyphs,
    std::function<filament::math::float3(mochi::Real3 const&)> const& toRenderer,
    filament::math::float4 color) {
  for (FrameAxisGlyph const& glyph : glyphs) {
    filament::math::float3 const tail = toRenderer(glyph.origin);
    filament::math::float3 const tip = toRenderer(glyph.origin + glyph.vector);
    DrawArrow(
        debugDraw,
        tail,
        tip,
        static_cast<float>(kFrameAxisGlyphRadiusFraction) * length(tip - tail),
        color,
        /*overlay=*/true);
  }
}

} // namespace superdex::studio
