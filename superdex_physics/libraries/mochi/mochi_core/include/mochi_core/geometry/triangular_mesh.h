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

#include <mochi_core/geometry/simplicial_mesh.h>
#include <mochi_core/utils/graph.h>
#include <mochi_core/utils/nd_array_utils.h>
#include <mochi_core/utils/transform_srt.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Forwards
namespace picojson {
class value;
}

namespace mochi {

/*
  Half-edge data
*/
struct HalfEdge {
  int edge = -1;
  int node = -1;
  int face = -1;
  int next = -1;
  int pair = -1;
};

/*
  Half-edge data structure
*/
struct HalfEdgeStructure {
  std::vector<HalfEdge> halfEdges;
  std::vector<Int2> edge2halfs;
  std::vector<int> node2half;
  std::vector<int> face2half;
};

/**
  Triangular Mesh Class
*/
class TriangularMesh final : public SimplicialMesh {
 public:
  static constexpr int kNodesPerElement = 3;

  // Construct a TriangularMesh and copy the data
  TriangularMesh(Span<Real3 const> coordinatesIn, Span<Int3 const> connectivityIn);

  // Helper for unit test that synthesis the pair of inputs
  TriangularMesh(std::pair<std::vector<Real3>, std::vector<Int3>> const& pair)
      : TriangularMesh(MakeSpan(pair.first), MakeSpan(pair.second)) {}

  // Get the node indices for each element
  Span<Int3 const> GetElementConnectivity() const;

  // Get the unit normal for each element
  Span<Real3 const> GetElementNormals() const;

  // Computes the node coordinates with the given displacements.
  void ComputeNodeCoordinates(Span<Real3 const> nodeDisplacements, Span<Real3> outNodeCoordinates)
      const;

  // Get the unit normal for each element given a different set of node coordinates.
  void ComputeElementNormals(Span<Real3 const> nodeCoordinates, Span<Real3> outElementNormals)
      const;

  // Compute angle-weighted normals at mesh nodes (half-edge structure required)
  void ComputeNodeNormals(Span<Real3> outNodeNormals) const;
  void ComputeNodeNormals(
      Span<Real3 const> nodeCoordinates,
      Span<Real3 const> elementNormals,
      Span<Real3> outNodeNormals) const;

  // Compute averaged normals at the mesh edges (half-edge structure required)
  void ComputeEdgeNormals(HalfEdgeStructure const& halfEdge, Span<Real3> outEdgeNormals) const;
  void ComputeEdgeNormals(
      HalfEdgeStructure const& halfEdge,
      Span<Real3 const> elementNormals,
      Span<Real3> outEdgeNormals) const;

  // Generate half-edge structure data
  HalfEdgeStructure GenerateHalfEdgeStructure() const;

  // The first Graph returned is the element-to-node connectivity graph for an expanded stencil
  // consisting of a triangle and up to three face neighbors. The second Graph gives the position of
  // each local node within a fixed-size 6-node stencil, where positions 0, 1, 2 are the nodes of
  // the triangle and positions 3, 4, 5 are nodes opposite edges with local indices 1->2, 2->0, 0->1
  // (some of which may be missing at boundaries).  E.g., if a triangle with element index 3 has
  // global nodes 7, 8, 9, and a single neighbor with global index 10 opposite edge with global
  // indices 7->8, we would have connectivity[3] = {7, 8, 9, 10} and stencil[3] = {0, 1, 2, 5}. With
  // positional placeholders, the corresponding fixed-size stencil would have global nodes {7, 8, 9,
  // _, _, 10}.
  std::pair<Graph<int, int>, Graph<int, int>> GenerateBendingConnectivityAndStencil() const;

 private:
  void EnsureOutwardNormals();

  void CheckMeshFlux() const;

  std::vector<Real3> _elementNormals;
};

/*********************************************************************************
  Inlines
*/

inline Span<Int3 const> TriangularMesh::GetElementConnectivity() const {
  return Unflatten<Int3 const>(GetFlatConnectivity());
}

inline Span<Real3 const> TriangularMesh::GetElementNormals() const {
  return {_elementNormals};
}

/*********************************************************************************
  Utility Functions
*/

/**
  Check if a triangle mesh is closed by testing unpaired half-edges
*/
bool IsMeshClosed(HalfEdgeStructure const& halfEdge);

/**
  Check if a triangle mesh is closed given a mapping from edges to their adjacent elements.
 */
bool IsMeshClosed(std::unordered_map<Int2, std::vector<int>, Int2Hash> const& edgeToElements);

/**
  Create a triangle mesh by clustering vertices within a threshold (measured using L-infinity
  norm). The method returns null if no clustering is performed or if the result is degenerate.
*/
std::shared_ptr<TriangularMesh const> TriangularMeshFromClusteredVertices(
    Span<Real3 const> coordinates,
    Span<Int3 const> connectivity,
    real distThreshold = 1e-6_r);

// Generate a mapping from edges (represented as sorted pairs of node indices) to lists of
// elements adjacent to them from a connectivity array.
std::unordered_map<Int2, std::vector<int>, Int2Hash> GenerateEdgeToElementsMap(
    Span<Int3 const> const& elementConnectivity);

/**
This is a utility function intended to be used in samples and unit tests, generating the nodes and
connectivity corresponding to a uniform mesh of a(n optionally scaled) unit square.  It is defined
outside of the respective sample and test utility/helper headers to be accessible in both places. It
produces a mesh of n[0] x n[1] cells, each split into two triangular elements, scaled by scale[0]
and scale[1] in the respective directions.  The axis parameter determines which spatial direction
the plane of the mesh is orthogonal to, such that n[i] and scale[i] apply to the spatial direction
(axis + i + 1) % 3.
 */
std::pair<std::vector<Real3>, std::vector<Int3>>
UniformSquareTriangularMeshData(Int2 n, Real2 scale = Real2{1_r, 1_r}, int axis = 2);
} // namespace mochi
