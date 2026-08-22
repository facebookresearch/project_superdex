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
#include <mochi_core/geometry/model_data.h>
#include <mochi_core/utils/error.h>

namespace mochi::mesh {

/// @brief Reconstruct a triangle surface mesh from a grid SDF using Marching Cubes.
///
/// @param[in] sdfData SDF grid data. `dims[i]` is the number of samples along axis i
///                    (corners of the grid, NOT cell count). Sample 0 sits at
///                    `bounds.min[i]`, sample `dims[i]-1` sits at `bounds.max[i]`.
///                    Mochi convention: negative inside surface, positive outside;
///                    iso-value 0 produces the surface.
/// @param[in,out] error Error status
/// @return Triangle mesh (nodesPerElement == 3) representing the zero-isosurface
[[nodiscard]] MeshData ReconstructSurfaceFromSdf(GridSdfDataView const& sdfData, Error& error);

} // namespace mochi::mesh
