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

#include "rendering/rod_visualization.h"

#include <mochi_core/utils/span.h>

#include <math/vec3.h>
#include <math/vec4.h>

#include <functional>

namespace mochi_renderer {
class DebugDraw;
} // namespace mochi_renderer

namespace superdex::studio {

// Arrowhead proportions of DrawArrow, as multiples of the shaft radius.
inline constexpr float kArrowHeadLength = 5.0f;
inline constexpr float kArrowHeadRadius = 2.2f;

// Draws a solid straight arrow from `tail` to `tip` (renderer space). The head keeps its
// proportions; an arrow too short for a full head shrinks as a whole so the head never exceeds
// half its length. `overlay` draws it on top of everything (no depth testing).
void DrawArrow(
    mochi_renderer::DebugDraw& debugDraw,
    filament::math::float3 tail,
    filament::math::float3 tip,
    float shaftRadius,
    filament::math::float4 color,
    bool overlay);

// Draws rod frame-axis glyphs as arrows on top of everything, each with a shaft radius of
// kFrameAxisGlyphRadiusFraction of its length. @p toRenderer maps a glyph point to renderer space.
void DrawFrameAxisGlyphs(
    mochi_renderer::DebugDraw& debugDraw,
    mochi::Span<FrameAxisGlyph const> glyphs,
    std::function<filament::math::float3(mochi::Real3 const&)> const& toRenderer,
    filament::math::float4 color);

} // namespace superdex::studio
