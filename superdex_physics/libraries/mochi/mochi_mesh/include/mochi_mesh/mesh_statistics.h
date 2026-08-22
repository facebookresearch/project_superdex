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

namespace mochi::mesh {

/// @brief [Experimental] Distribution statistics (mean, standard deviation, min, max).
/// @warning This API may change in future releases.
struct DistributionStatistics {
  double mean = 0;
  double standardDeviation = 0;
  double min = 0;
  double max = 0;
};

/// @brief [Experimental] Quality statistics for a triangle surface mesh.
/// @warning This API may change in future releases.
struct MeshStatistics {
  int numVertices = 0;
  int numFaces = 0;
  DistributionStatistics edgeLengths; ///< Edge length statistics [coordinate units]
  DistributionStatistics angles; ///< Interior angle statistics [degrees]
  double hausdorffDistance = -1; ///< Approximate Hausdorff distance to reference mesh [coordinate
                                 ///< units], -1 if not computed
  bool isClosed = false; ///< Whether the mesh is a watertight 2-manifold: every undirected edge is
                         ///< shared by exactly two triangles (boundary edges or non-manifold edges
                         ///< shared by 3+ triangles both make this false).
};

/// @brief [Experimental] Compute quality statistics for a triangle surface mesh.
/// @warning This API may change in future releases.
///
/// @param[in] mesh Triangle mesh to analyze (nodesPerElement must be 3)
/// @param[in] referenceMesh Optional reference mesh for Hausdorff distance (nullptr to skip)
/// @param[in,out] error Error status
/// @return Mesh quality statistics
[[nodiscard]] MeshStatistics
ComputeMeshStatistics(MeshDataView const& mesh, MeshDataView const* referenceMesh, Error& error);

} // namespace mochi::mesh
