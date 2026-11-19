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

#include "rendering/rod_visualization.h"

#include <mochi_core/geometry/mesh_data_utils.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/nd_array_utils.h>

namespace superdex::studio {

using mochi::Int2;
using mochi::operator""_r;
using mochi::real;
using mochi::Real3;

std::vector<Int2> PolylineElements(int numNodes, mochi::Span<int const> connectivity) {
  if (!connectivity.empty()) {
    auto const elements = mochi::Unflatten<Int2 const>(connectivity);
    return {elements.begin(), elements.end()};
  }
  if (numNodes < 2) {
    return {};
  }
  mochi::DynamicArray<int> const sequential =
      mochi::MakeSequentialPolylineConnectivity(numNodes, /*isClosedLoop=*/false);
  auto const elements = mochi::Unflatten<Int2 const>(mochi::MakeConstSpan(sequential));
  return {elements.begin(), elements.end()};
}

std::vector<FrameAxisGlyph> ComputeFrameAxisGlyphs(
    mochi::Span<real const> nodeCoordinates,
    mochi::Span<int const> connectivity,
    mochi::Span<real const> frameAxes) {
  auto const nodes = mochi::Unflatten<Real3 const>(nodeCoordinates);
  std::vector<Int2> const elements = PolylineElements(mochi::isize(nodes), connectivity);
  if (elements.empty() || mochi::isize(frameAxes) != 3 * mochi::isize(elements)) {
    return {};
  }
  auto const axes = mochi::Unflatten<Real3 const>(frameAxes);
  real totalLength = 0_r;
  for (Int2 const& element : elements) {
    totalLength += mochi::Norm(nodes[element[1]] - nodes[element[0]]);
  }
  real const glyphLength =
      kFrameAxisGlyphLengthFraction * totalLength / static_cast<real>(elements.size());
  std::vector<FrameAxisGlyph> glyphs;
  glyphs.reserve(elements.size());
  for (int i = 0; i < mochi::isize(elements); ++i) {
    Real3 const midpoint = (nodes[elements[i][0]] + nodes[elements[i][1]]) * 0.5_r;
    glyphs.push_back({midpoint, axes[i] * glyphLength});
  }
  return glyphs;
}

} // namespace superdex::studio
