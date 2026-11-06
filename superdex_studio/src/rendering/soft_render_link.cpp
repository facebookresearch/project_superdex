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

#include "rendering/soft_render_link.h"

#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/coordinate_space_converter.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/string_utils.h>
#include <mochi_renderer/deformable_mesh.h>
#include <mochi_renderer/render_space.h>

#include <algorithm>

namespace superdex::studio {

RenderModelLink CheckRenderModelLink(
    mochi_renderer::DeformableMeshView renderMesh,
    mochi::MeshDataView const& visualMesh,
    ModelPlacement const& render,
    ModelPlacement const& shape) {
  using namespace mochi;
  if (visualMesh.IsEmpty()) {
    return {.reason = "The Mochi model has no visual mesh."};
  }
  if (!visualMesh.skinning.has_value()) {
    return {.reason = "The visual mesh has no deformation embedding."};
  }
  int const renderVertexCount = isize(renderMesh.restPositions) / 3;
  if (renderVertexCount != visualMesh.GetNumNodes()) {
    return {
        .reason = Format(
            "The render model has %d vertices, but the visual mesh has %d.",
            renderVertexCount,
            visualMesh.GetNumNodes())};
  }
  if (!std::equal(
          renderMesh.triangleIndices.begin(),
          renderMesh.triangleIndices.end(),
          visualMesh.connectivity.begin(),
          visualMesh.connectivity.end())) {
    return {.reason = "The render model's triangles differ from the visual mesh's."};
  }

  // Both placed in the actor frame, in Mochi's axes.
  CoordinateSpaceConverter const renderToMochi(
      mochi_renderer::RenderSpace(), CoordinateSpace::Default());
  // Bounds the size of a placed point and of each term placing it; float round-off scales with it.
  auto const placedMagnitude = [](ModelPlacement const& placement, Real3 const& point) {
    return Norm(placement.scale) * Norm(point) + Norm(placement.transform.GetTranslation());
  };
  real maxDistance = 0_r;
  real magnitude = 0_r;
  for (int node = 0; node < renderVertexCount; ++node) {
    Real3 const renderSource = renderToMochi.TranslationToOutput(
        Real3{
            renderMesh.restPositions[3 * node],
            renderMesh.restPositions[3 * node + 1],
            renderMesh.restPositions[3 * node + 2]});
    Real3 const visualSource{
        visualMesh.coordinates[3 * node],
        visualMesh.coordinates[3 * node + 1],
        visualMesh.coordinates[3 * node + 2]};
    maxDistance = Max(maxDistance, Norm(render.Apply(renderSource) - shape.Apply(visualSource)));
    magnitude =
        Max(magnitude,
            Max(placedMagnitude(render, renderSource), placedMagnitude(shape, visualSource)));
  }
  // Relative to that size, so round-off passes wherever and however large the part is.
  real const tolerance = 1e-4_r * magnitude;
  if (maxDistance > tolerance) {
    return {
        .reason = Format(
            "The render model is up to %.3g m from the visual mesh: its offset or scale differs "
            "from the Mochi model's, or either file changed after embedding.",
            static_cast<double>(maxDistance))};
  }
  return {.linked = true};
}

} // namespace superdex::studio
