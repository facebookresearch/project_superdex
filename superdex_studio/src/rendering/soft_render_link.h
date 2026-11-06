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

#include <mochi_core/geometry/mesh_data.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/transform_rt.h>

#include <string>

namespace mochi_renderer {
struct DeformableMeshView;
} // namespace mochi_renderer

namespace superdex::studio {

// Where a model sits in its actor's frame: scaled per axis, then rotated and translated -- the
// order mochi::model_utils::BakeTransform applies a soft actor's scale and shape offset in. Points
// are in Mochi's axes; a render model's are its file's, converted from renderer axes.
struct ModelPlacement {
  mochi::Real3 scale = {mochi::real(1), mochi::real(1), mochi::real(1)};
  mochi::TransformRT transform = {};

  [[nodiscard]] mochi::Real3 Apply(mochi::Real3 const& point) const {
    return transform.TransformPoint(scale * point);
  }
};

// Whether a soft actor's render model can be drawn deformed by its visual mesh, and if not, why.
struct RenderModelLink {
  bool linked = false;
  std::string reason; // empty when linked
};

// Whether @p renderMesh (renderer axes) is the visual mesh @p visualMesh (Mochi axes, model frame),
// as the Model Editor embedded it: the same vertices and triangle indices, and, placed by @p
// render, where the visual mesh is when placed by @p shape. Only then does each simulated
// visual-mesh node position a render-model vertex.
RenderModelLink CheckRenderModelLink(
    mochi_renderer::DeformableMeshView renderMesh,
    mochi::MeshDataView const& visualMesh,
    ModelPlacement const& render,
    ModelPlacement const& shape);

} // namespace superdex::studio
