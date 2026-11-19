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

// Rod (polyline) visualization helpers, and the pure geometry of picking and grabbing deformables.
// No renderer, so it runs anywhere.

#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/span.h>

#include <optional>
#include <string_view>
#include <vector>

namespace superdex::studio {

// Role suffix hashed with a model's color seed to color its frame-axis arrows, so they read alike
// everywhere they are shown.
inline constexpr std::string_view kFrameAxesColorRole = " frame axes";

// A rod element's frame axis, drawn as an arrow from `origin` along `vector`.
struct FrameAxisGlyph {
  mochi::Real3 origin;
  mochi::Real3 vector;
};

// Frame-axis arrow length as a fraction of the rod's mean element length.
inline constexpr mochi::real kFrameAxisGlyphLengthFraction = static_cast<mochi::real>(0.5);
// Frame-axis arrow shaft radius as a fraction of the arrow length.
inline constexpr mochi::real kFrameAxisGlyphRadiusFraction = static_cast<mochi::real>(0.04);

// A rod's deformed centerline nodes and element frame axes in its local frame (Mochi space), 3
// values per node and per element.
struct RodPose {
  std::vector<mochi::real> nodes;
  std::vector<mochi::real> frameAxes;
};

// The node pairs of a polyline's elements: `connectivity` itself, or for an empty one the open
// polyline through its `numNodes` nodes in order.
std::vector<mochi::Int2> PolylineElements(int numNodes, mochi::Span<int const> connectivity);

// One glyph per polyline element, from the element's midpoint along its frame axis (3 values per
// element in `frameAxes`). Every glyph is kFrameAxisGlyphLengthFraction of the polyline's mean
// element length, so they suit the discretization whatever the model's units. Empty if
// `frameAxes` does not hold one axis per element.
std::vector<FrameAxisGlyph> ComputeFrameAxisGlyphs(
    mochi::Span<mochi::real const> nodeCoordinates,
    mochi::Span<int const> connectivity,
    mochi::Span<mochi::real const> frameAxes);

// The nodes a grab at @p grabPoint springs, nearest first: the nearest node (the anchor, first)
// and every node within @p radiusFraction of the nodes' bounding-box diagonal, at most @p maxNodes.
// Empty when there are no nodes.
std::vector<int> SelectGrabCluster(
    mochi::Span<mochi::Real3 const> nodes,
    mochi::Real3 const& grabPoint,
    mochi::real radiusFraction,
    int maxNodes);

// A polyline segment before perspective division, using the camera's OpenGL-style rendering
// projection.
struct ClipSegment {
  mochi::Float4 a;
  mochi::Float4 b;
};
struct ScreenSegmentHit {
  int segment = -1;
  // World-space interpolation parameter of the original segment, 0 at `a` to 1 at `b`.
  float t = 0;
  float distancePx = 0;
};

// Picks the segment in front of the near plane nearest @p cursor in framebuffer pixels,
// within @p maxDistancePx.
// Equal screen distances are broken by depth, then by segment order.
std::optional<ScreenSegmentHit> PickNearestScreenSegment(
    mochi::Span<ClipSegment const> segments,
    mochi::Float2 cursor,
    mochi::Float2 viewportSize,
    float maxDistancePx);

} // namespace superdex::studio
