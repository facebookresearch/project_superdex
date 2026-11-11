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
#include <mochi_core/utils/error.h>
#include <mochi_mesh/mochi_mesh_cli_types.h>

namespace mochi::mesh {

/// @brief [Experimental] Generate a tetrahedral volume mesh from a closed triangle surface.
/// @warning This API may change in future releases.
///
/// @details Every output node is referenced by at least one tetrahedron, so the result can be used
/// directly as a soft actor shape. The output is deterministic for a given input and parameters.
///
/// @param[in] surfaceMesh Input triangle mesh (nodesPerElement must be 3). Must be closed and free
/// of self-intersections.
/// @param[in] params Tetrahedralization parameters
/// @param[in,out] error Error status
/// @return Tetrahedral mesh (nodesPerElement == 4)
[[nodiscard]] MeshData Tetrahedralize(
    MeshDataView const& surfaceMesh,
    TetrahedralizationParams const& params,
    Error& error);

/// @brief [Experimental] Generate a tetrahedral volume mesh whose boundary is a closed triangle
/// surface.
/// @warning This API may change in future releases.
///
/// @details Unlike @ref Tetrahedralize, which approximates the input surface, this keeps it: the
/// first nodes of the output are the input vertices, unmoved and in input order, followed by any
/// added nodes. Some input triangles may be subdivided by nodes added on their edges or faces,
/// which keeps the boundary's shape; in strict mode that fails instead, so the output's boundary
/// triangles are exactly the input triangles. Nested shells alternate between material and empty
/// cavities, starting with material inside the outermost shells. Because the boundary is kept,
/// the quality of the tetrahedra next to it is limited by the quality of the
/// input triangles. Every output node is referenced by at least one tetrahedron, and the output is
/// deterministic for a given input and parameters.
///
/// @param[in] surfaceMesh Input triangle mesh (nodesPerElement must be 3). Must be a closed,
/// consistently oriented manifold free of self-intersections, with no unreferenced or coincident
/// vertices.
/// @param[in] params Tetrahedralization parameters
/// @param[in,out] error Error status
/// @return Tetrahedral mesh (nodesPerElement == 4)
[[nodiscard]] MeshData TetrahedralizeConstrained(
    MeshDataView const& surfaceMesh,
    ConstrainedTetrahedralizationParams const& params,
    Error& error);

} // namespace mochi::mesh
