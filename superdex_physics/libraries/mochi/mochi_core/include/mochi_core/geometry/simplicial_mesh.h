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

#include <mochi_core/geometry/aabb.h>
#include <mochi_core/geometry/obb.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/span.h>

#include <unordered_map>
#include <vector>

namespace mochi {

/**
  SimplicialMesh

  A base class for meshes composed of simplicial elements (eg. point, segments,triangles,
  tetrahedra ect)

  Terminology:
    Node = A 3D point in the mesh (also referred to as a "vertex" or "coordinate")
    Element = A simplex formed by connected nodes
    Edge = A line segment connecting two nodes on an element
    Active Nodes = nodes used in the connectivity and topology of the mesh. Ordinary accessors
      return the full stored mesh, which may include nodes unused by any element connectivity.
      Accessors for the active-node mesh, containing only nodes referenced by connectivity,
      contain the word Active in the method name.
*/
class SimplicialMesh {
 public:
  static constexpr int kSpaceDimension = 3;

  SimplicialMesh(
      Span<Real3 const> const& nodeCoordinates,
      Span<int const> const& elementConnectivity,
      int nodesPerElement);

  virtual ~SimplicialMesh() = default;

  // Get the number of nodes (e.g. vertices)
  int GetNumNodes() const;

  // Get the number of nodes used in the connectivity (e.g. vertices)
  int GetNumActiveNodes() const;

  // Get the number of unique edges (1D segments formed by connected nodes)
  int GetNumEdges() const;

  // Get the number of unique faces (2D surfaces formed by connected nodes).
  // For a TriangularMesh this is the same as GetNumElements().
  // For a TetrahedralMesh this is the triangular faces of the elements.
  int GetNumFaces() const;

  // Get the number of unique volumes (3D shapes formed by connected nodes).
  // For a TriangleMesh this is always zero.
  // For a TetrahedralMesh this is the same as GetNumElements().
  int GetNumVolumes() const;

  // Get the number of elements (triangles or tetrahedra)
  int GetNumElements() const;

  // Get the number of nodes per face (3 for triangular faces)
  int GetNumNodesPerFace() const;

  // Get the number of nodes per element (3 for triangles, 4 for tetrahedra)
  int GetNumNodesPerElement() const;

  // Get the coordinates of each node
  // @TODO[MAURIZIO]: allow to access only a portion of the coordinates
  // @TODO[MAURIZIO]: relabel `NodesCoordinates` as we are getting multiple nodes
  Span<Real3 const> GetNodeCoordinates() const;

  // Get the coordinates of active nodes. The order is the ascending node-index order returned by
  // GetActiveNodes().
  Span<Real3 const> GetActiveNodeCoordinates() const;

  // Get a flat array of node indices. Every group of GetNumNodesPerElement()
  // indices represent an element.
  Span<int const> GetFlatConnectivity() const;

  // Generate unique list of nodes in a given list of elements.
  void UniqueNodesInElements(Span<int const> elementIndices, std::vector<int>& outNodes) const {
    // NOTE: Performs dynamic memory allocation but cost seems negligible.
    std::vector<bool> seenNodes(GetNumNodes(), false);
    outNodes.clear();
    outNodes.reserve(elementIndices.size() * _numNodesPerElement);
    Span<int const> connec = GetFlatConnectivity();
    for (int elemIdx : elementIndices) {
      MOCHI_ASSERT_VERBOSE(elemIdx >= 0 && elemIdx < GetNumElements(), "Element out of range.");
      auto const startAt = elemIdx * _numNodesPerElement;
      for (int i = 0; i < _numNodesPerElement; ++i) {
        int const nodeIdx = connec[startAt + i];
        if (!seenNodes[nodeIdx]) {
          outNodes.push_back(nodeIdx);
          seenNodes[nodeIdx] = true;
        }
      }
    }
  }

  // Get a flat array of node indices into GetActiveNodeCoordinates().
  // This is GetFlatConnectivity() remapped with GetAllToActiveNodesIndexMap(). If there are n
  // active nodes, indices in this connectivity are in [0, n).
  Span<int const> GetActiveNodesFlatConnectivity() const;

  // Get the indices of all unique nodes on the boundary
  Span<int const> GetBoundaryNodes() const;

  // Get the indices of all active nodes in ascending full-mesh node-index order.
  // Active nodes are the subset of nodes used in the connectivity. If there are no unreferenced
  // nodes, active nodes are the same as the full set of nodes.
  Span<int const> GetActiveNodes() const;

  // Get all unique edges as pairs of full-mesh node indices.
  Span<Int2 const> GetEdges() const;

  // Get all unique edges as pairs of active-node indices, similarly to
  // GetActiveNodesFlatConnectivity().
  Span<Int2 const> GetActiveNodesEdges() const;

  // Get just the edges on the boundary (edges connected by exactly one element)
  Span<Int2 const> GetBoundaryEdges() const;

  // Get the overall barycenter of the mesh
  Real3 const& GetBarycenter() const;

  // Get the barycenter of a specific element.
  Real3 const& GetElementBarycenter(int elementIndex) const;

  // Get the barycenter of a specific element.
  Span<Real3 const> GetElementsBarycenters() const;

  // Given a node index in the mesh list of nodes finds the index of the
  // active node
  int GetAllToActiveNodesIndexMap(int i) const;

  // Given a node index in the active list of nodes finds the index of the
  // node in the full set of nodes
  int GetActiveToAllNodesIndexMap(int i) const;

  // Get the "measure" (area of triangle or volume of tetrahedron) of a single element
  real GetElementMeasure(int elementIndex) const;

  // Get the total "measure" (area of triangle or volume of tetrahedron) for the whole mesh
  real GetTotalMeasure() const;

  // Get the axis-aligned bounding box that contains all the nodes
  Aabb const& GetAabb() const;

  // Get an oriented bounding box that contains all the nodes
  Obb GetObb() const;

  inline Span<int const> GetAdjacentElements(int nodeIndex) const;

 protected:
  // Static helpers
  static std::vector<int> FindBoundaryNodes(size_t numNodes, Span<Int2 const> boundaryEdges);
  static std::vector<int> BuildSimplicialMeshAdjacency(
      size_t numNodes,
      Span<int const> connectivity,
      int numNodesPerElement);

  std::vector<Real3> _coordinates;
  std::vector<Real3> _coordinatesActiveNodes;
  std::vector<int> _connectivity;
  std::vector<int> _connectivityActiveNodes;
  int const _numNodesPerElement;
  std::vector<Real3> _elementBarycenters;
  Aabb _aabb;

  // Computed by the derived class:
  std::vector<Int2> _edges;
  std::vector<Int2> _edgesActiveNodes;
  std::vector<Int2> _boundaryEdges;
  std::vector<int> _boundaryNodes;
  std::vector<real> _elementMeasure; // area or volume
  std::vector<int> _nodeElementAdjacency;
  std::vector<int> _activeNodes; // All the nodes used in the connectivity
  std::unordered_map<int, int> _allToActiveMap;

  Real3 _barycenter = {};
  real _totalMeasure = 0_r;
  int _numFaces = 0;
  int _numNodesPerFace = 0;
  int _numVolumes = 0;
};

/*********************************************************************************
  Inlines
*/

inline int SimplicialMesh::GetNumNodes() const {
  return isize(_coordinates);
}

inline int SimplicialMesh::GetNumActiveNodes() const {
  return isize(_activeNodes);
}

inline int SimplicialMesh::GetNumEdges() const {
  return isize(_edges);
}

inline int SimplicialMesh::GetNumFaces() const {
  return _numFaces;
}

inline int SimplicialMesh::GetNumVolumes() const {
  return _numVolumes;
}

inline int SimplicialMesh::GetNumElements() const {
  return isize(_connectivity) / _numNodesPerElement;
}

inline int SimplicialMesh::GetNumNodesPerFace() const {
  return _numNodesPerFace;
}

inline int SimplicialMesh::GetNumNodesPerElement() const {
  return _numNodesPerElement;
}

inline Span<Real3 const> SimplicialMesh::GetNodeCoordinates() const {
  return {_coordinates};
}

inline Span<Real3 const> SimplicialMesh::GetActiveNodeCoordinates() const {
  return {_coordinatesActiveNodes};
}
inline Span<int const> SimplicialMesh::GetBoundaryNodes() const {
  return {_boundaryNodes};
}

inline Span<int const> SimplicialMesh::GetActiveNodes() const {
  return {_activeNodes};
}

inline Span<int const> SimplicialMesh::GetFlatConnectivity() const {
  return {_connectivity};
}

inline Span<int const> SimplicialMesh::GetActiveNodesFlatConnectivity() const {
  return {_connectivityActiveNodes};
}

inline Span<Int2 const> SimplicialMesh::GetEdges() const {
  return {_edges};
}

inline Span<Int2 const> SimplicialMesh::GetActiveNodesEdges() const {
  return {_edgesActiveNodes};
}

inline int SimplicialMesh::GetAllToActiveNodesIndexMap(int i) const {
  return _allToActiveMap.at(i);
}

inline int SimplicialMesh::GetActiveToAllNodesIndexMap(int i) const {
  return _activeNodes[i];
}

inline Real3 const& SimplicialMesh::GetBarycenter() const {
  return _barycenter;
}

inline Span<Real3 const> SimplicialMesh::GetElementsBarycenters() const {
  return {_elementBarycenters};
}

inline Span<Int2 const> SimplicialMesh::GetBoundaryEdges() const {
  return {_boundaryEdges};
}

inline real SimplicialMesh::GetElementMeasure(int elementIndex) const {
  return _elementMeasure[elementIndex];
}

inline real SimplicialMesh::GetTotalMeasure() const {
  return _totalMeasure;
}

inline Aabb const& SimplicialMesh::GetAabb() const {
  return _aabb;
}

inline Real3 const& SimplicialMesh::GetElementBarycenter(int elementIndex) const {
  // NOTE: We could expose _elementBarycenters as a Span, but this function signature
  // gives us the flexibility to compute the barycenters on-the-fly if desired.
  return _elementBarycenters[elementIndex];
}

// Utility function
inline Span<int const> GetAdjacentIndices(int nodeIndex, Span<int const> adjacency) {
  // adjacency starts with a table of offsets. The spans of adjacent node/face
  // indices are stored later in the same array.
  int const* ptr = &adjacency[adjacency[nodeIndex]];
  size_t const len = adjacency[nodeIndex + 1] - adjacency[nodeIndex];
  return {ptr, len};
}

inline Span<int const> SimplicialMesh::GetAdjacentElements(int nodeIndex) const {
  return mochi::GetAdjacentIndices(nodeIndex, _nodeElementAdjacency);
}

} // namespace mochi
