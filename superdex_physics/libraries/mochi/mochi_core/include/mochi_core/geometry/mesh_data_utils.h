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
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/span.h>

#include <algorithm>

namespace mochi {

/**
 * @brief Build a sequential polyline connectivity array.
 *
 * For an open polyline (numNodes-1 segments), the result is
 *   [0,1, 1,2, ..., numNodes-2, numNodes-1].
 * For a closed-loop polyline (numNodes segments), a wrap-around segment
 *   [numNodes-1, 0]
 * is appended.
 *
 * @param[in] numNodes Number of nodes in the polyline (>= 2; >= 3 if closed loop).
 * @param[in] isClosedLoop If true, include the wrap-around segment.
 * @return Flat connectivity array of size 2 * (numNodes - !isClosedLoop).
 */
inline DynamicArray<int> MakeSequentialPolylineConnectivity(int numNodes, bool isClosedLoop) {
  MOCHI_ASSERT(numNodes >= 2, "Polyline must have at least 2 nodes");
  MOCHI_ASSERT(!isClosedLoop || numNodes >= 3, "Closed-loop polyline must have at least 3 nodes");
  int const numSegments = isClosedLoop ? numNodes : numNodes - 1;
  DynamicArray<int> connectivity;
  connectivity.reserve(2 * numSegments);
  for (int i = 0; i < numSegments; ++i) {
    connectivity.push_back(i);
    connectivity.push_back((i + 1) % numNodes);
  }
  return connectivity;
}

/**
 * @brief Detect whether a polyline mesh encodes a closed loop.
 *
 * @details A polyline mesh (@ref MeshDataView::nodesPerElement == 2) is treated as a closed loop
 * when it has an explicit wrap-around segment, i.e. the connectivity array contains 2 * numNodes
 * entries (one more segment than the open case, with the last segment wrapping numNodes-1 -> 0).
 * Open polylines have 2 * (numNodes - 1) entries. An empty connectivity array is treated as open.
 *
 * @param[in] mesh Mesh data view to inspect.
 * @return True iff @p mesh is a polyline (@ref MeshDataView::nodesPerElement == 2) and its
 * connectivity size equals 2 * numNodes.
 */
[[nodiscard]] inline bool IsPolylineClosedLoop(MeshDataView const& mesh) {
  return mesh.nodesPerElement == 2 && !mesh.connectivity.empty() &&
      (isize(mesh.connectivity) == 2 * mesh.GetNumNodes());
}

/**
 * @brief Compute unique edge indices from mesh connectivity.
 *
 * @details Extracts all unique edges from the element connectivity. Each edge is a pair of node
 * indices (min, max) sorted lexicographically.
 *
 * @param[in] mesh Mesh data view with valid connectivity and nodesPerElement >= 2.
 *
 * @return Flat array of edge node indices [a0, b0, a1, b1, ...] where a < b. Size is
 * numUniqueEdges * 2.
 */
inline DynamicArray<int> ComputeEdgeIndices(MeshDataView const& mesh) {
  if (mesh.connectivity.empty() || mesh.nodesPerElement < 2) {
    return {};
  }
  int const numElements = mesh.GetNumElements();
  int const npe = mesh.nodesPerElement;
  int const edgesPerElement = npe * (npe - 1) / 2;
  Span<int const> const& conn = mesh.connectivity;

  DynamicArray<int64_t> edges;
  edges.reserve(numElements * edgesPerElement);
  for (int e = 0; e < numElements; ++e) {
    int const base = e * npe;
    for (int i = 0; i < npe; ++i) {
      for (int j = i + 1; j < npe; ++j) {
        int a = conn[base + i];
        int b = conn[base + j];
        if (a > b) {
          std::swap(a, b);
        }
        edges.push_back((static_cast<int64_t>(a) << 32) | b);
      }
    }
  }
  std::sort(edges.begin(), edges.end());
  edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

  DynamicArray<int> result;
  result.resize_noinit(isize(edges) * 2);
  for (int i = 0; i < isize(edges); ++i) {
    result[i * 2 + 0] = static_cast<int>(edges[i] >> 32);
    result[i * 2 + 1] = static_cast<int>(edges[i] & 0xFFFFFFFF);
  }
  return result;
}

/**
 * @brief Compute element barycenters from mesh coordinates and connectivity.
 *
 * @details For each element, computes the average of its node positions.
 *
 * @param[in] mesh Mesh data view with valid coordinates, connectivity, and nodesPerElement.
 *
 * @return Flat array of barycenter coordinates [x0, y0, z0, x1, y1, z1, ...] with size
 * numElements * 3.
 */
inline DynamicArray<real> ComputeElementBarycenters(MeshDataView const& mesh) {
  if (mesh.connectivity.empty() || mesh.nodesPerElement < 1) {
    return {};
  }

  int const numElements = mesh.GetNumElements();
  Span<int const> const& conn = mesh.connectivity;
  Span<real const> const& coords = mesh.coordinates;
  real const invNodesPerElement = 1_r / static_cast<real>(mesh.nodesPerElement);

  DynamicArray<real> result;
  result.resize_noinit(numElements * kMeshDataSpaceDim);

  for (int e = 0; e < numElements; ++e) {
    int const base = e * mesh.nodesPerElement;
    real cx = 0_r;
    real cy = 0_r;
    real cz = 0_r;
    for (int i = 0; i < mesh.nodesPerElement; ++i) {
      int const nodeIdx = conn[base + i];
      cx += coords[nodeIdx * 3 + 0];
      cy += coords[nodeIdx * 3 + 1];
      cz += coords[nodeIdx * 3 + 2];
    }
    result[e * 3 + 0] = cx * invNodesPerElement;
    result[e * 3 + 1] = cy * invNodesPerElement;
    result[e * 3 + 2] = cz * invNodesPerElement;
  }
  return result;
}

} // namespace mochi
