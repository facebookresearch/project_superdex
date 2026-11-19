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

// Triangle and polyline geometry for a Mochi model, with no dependency on the Filament engine, so
// tools that only read models can use it without the renderer. SuperDex Studio's processing code
// reads Mochi models through it, which lets `superdex_studio --process` and the processing tests
// run on Linux CI and on devservers that have no display. Screenshots still render through the full
// renderer.

#include <vector>

namespace mochi {
struct MeshDataView;
struct ModelData;
class CoordinateSpaceConverter;
} // namespace mochi

namespace mochi_renderer {

// Extracts renderable triangle geometry from a @ref mochi::ModelData, converting it
// from Mochi space into the renderer's space (@ref RenderSpace).
//
// For a tetrahedral mesh the boundary surface is extracted; a triangle surface mesh is
// used directly; otherwise a procedural box/plane/sphere is generated from the model's
// analytic shape. Per-vertex normals are angle-weighted (mesh) or supplied by the
// generator (procedural). Positions/normals are transformed by `converter`; when
// `converter` is null a default Mochi-to-@ref RenderSpace
// @ref mochi::CoordinateSpaceConverter is used. Triangle winding is left unchanged.
//
// @return false (leaving the outputs unspecified) if the model has no usable geometry.
bool BuildMochiModelGeometry(
    mochi::ModelData const& modelData,
    mochi::CoordinateSpaceConverter const* converter,
    std::vector<float>& positions,
    std::vector<float>& normals,
    std::vector<int>& indices);

// Extracts the line segments of a valid polyline (rod) mesh (see @ref mochi::model::Validate),
// converting positions like @ref BuildMochiModelGeometry. `segmentIndices` holds two node indices
// per segment of the open or closed polyline through the nodes in order.
//
// @return false (leaving the outputs unspecified) if @p mesh is not a polyline mesh with at least
// two nodes.
bool BuildMochiPolylineGeometry(
    mochi::MeshDataView const& mesh,
    mochi::CoordinateSpaceConverter const* converter,
    std::vector<float>& positions,
    std::vector<int>& segmentIndices);

} // namespace mochi_renderer
