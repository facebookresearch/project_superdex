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

#include <mochi_core/geometry/geometry_utils.h>
#include <mochi_core/geometry/simplicial_mesh.h>
#include <mochi_core/utils/array_utils.h>
#include <mochi_core/utils/nd_array_utils.h>

#include <algorithm>
#include <numeric>
#include <set>
#include <unordered_set>
#include <vector>

namespace mochi {

// Compute the barycenter for each element in a simplicial mesh
static std::vector<Real3> CalcElementBarycenters(
    Span<Real3 const> coordinates,
    Span<int const> connectivity,
    int numNodesPerElement) {
  MOCHI_ASSERT_VERBOSE(
      connectivity.size() % numNodesPerElement == 0,
      "Connectivity array size must be a multiple of nodesPerElement");
  size_t const numElements = connectivity.size() / numNodesPerElement;
  std::vector<Real3> barycenters(numElements);
  for (size_t e = 0; e < numElements; ++e) {
    size_t const base = e * numNodesPerElement;
    Real3 sum = {};
    for (int i = 0; i < numNodesPerElement; ++i) {
      auto nodeIdx = connectivity[base + i];
      sum += coordinates[nodeIdx];
    }
    barycenters[e] = sum / static_cast<real>(numNodesPerElement);
  }
  return barycenters;
}

SimplicialMesh::SimplicialMesh(
    Span<Real3 const> const& nodeCoordinates,
    Span<int const> const& elementConnectivity,
    int numNodesPerElement)
    : _coordinates(nodeCoordinates.begin(), nodeCoordinates.end()),
      _connectivity(elementConnectivity.begin(), elementConnectivity.end()),
      _numNodesPerElement(numNodesPerElement) {
  MOCHI_ASSERT_VERBOSE(numNodesPerElement >= 2, "Unsupported simplex dimension");
  MOCHI_ASSERT_VERBOSE(
      elementConnectivity.size() % numNodesPerElement == 0,
      "Connectivity array size must be a multiple of numNodesPerElement");
  // Element Barycenters
  _elementBarycenters =
      CalcElementBarycenters(nodeCoordinates, elementConnectivity, numNodesPerElement);

  // Aabb
  _aabb = CalcAabb(_coordinates);

  // Build the adjacency list of all simplices  shared by a given node
  _nodeElementAdjacency =
      BuildSimplicialMeshAdjacency(_coordinates.size(), _connectivity, numNodesPerElement);

  // Set up arrays realted to active nodes
  // Active nodes are defined as the subset of nodes used in the connectivity
  std::set<int> activeNodesSet(elementConnectivity.begin(), elementConnectivity.end());
  _activeNodes = std::vector<int>(activeNodesSet.begin(), activeNodesSet.end());
  _coordinatesActiveNodes.reserve(_activeNodes.size());
  for (int i : _activeNodes) {
    _coordinatesActiveNodes.push_back(nodeCoordinates[i]);
    _allToActiveMap[i] = isize(_coordinatesActiveNodes) - 1;
  }
  _connectivityActiveNodes.reserve(_connectivity.size());
  for (int i : _connectivity) {
    _connectivityActiveNodes.push_back(_allToActiveMap[i]);
  }

  // The remaining protected data members are initialized by the derived class because different
  // algorithms are used depending on the element type.
}

std::vector<int> SimplicialMesh::FindBoundaryNodes(
    size_t numNodes,
    Span<Int2 const> boundaryEdges) {
  // Mark each node on the boundary
  std::vector<int> boundaryNodes(numNodes, -1);
  for (int i : Flatten(boundaryEdges)) {
    boundaryNodes[i] = i;
  }
  // Prune nodes that were not marked
  boundaryNodes.erase(
      std::remove(boundaryNodes.begin(), boundaryNodes.end(), -1), boundaryNodes.end());
  boundaryNodes.shrink_to_fit();

  MOCHI_ASSERT_VERBOSE(
      std::is_sorted(boundaryNodes.begin(), boundaryNodes.end()),
      "This algorithm should produce sorted boundary indicies.");

  return boundaryNodes;
}

std::vector<int> SimplicialMesh::BuildSimplicialMeshAdjacency(
    size_t numNodes,
    Span<int const> connectivity,
    int numNodesPerElement) {
  // If k = _numNodesPerElement, the mesh is comprised with k-simplices.
  // For each vertex build out the unique list of adjacent vertices.
  // adjacencySets maps a vertex index to all the adjacent vertices.
  std::vector<std::unordered_set<int>> adjacencySets;
  adjacencySets.resize(numNodes);
  for (int iSimplex = 0; iSimplex < isize(connectivity) / numNodesPerElement; ++iSimplex) {
    for (int iNode = 0; iNode < numNodesPerElement; ++iNode) {
      adjacencySets[connectivity[iSimplex * numNodesPerElement + iNode]].insert(iSimplex);
    }
  }

  // Calculate the space needed to store the list of sets in a flat vector
  size_t adjacencyDataSize = numNodes + 1; // +1 to store the end offset
  for (auto const& set : adjacencySets) {
    adjacencyDataSize += set.size();
  }

  // adjacency[i] is the offset of the start of the adjacency data for node i.
  // (adjacency[i+1] - adjacency[i]) is the number of adjacent face indices.
  std::vector<int> adjacency;
  adjacency.reserve(adjacencyDataSize);
  adjacency.resize(numNodes + 1);
  for (size_t i = 0; i < numNodes; ++i) {
    auto const& adjacentFaces = adjacencySets[i];
    adjacency[i] = isize(adjacency);
    adjacency.insert(adjacency.end(), adjacentFaces.begin(), adjacentFaces.end());
  }
  MOCHI_ASSERT_VERBOSE(
      adjacency.size() == adjacencyDataSize, "Adjacency table offsets are incorrect");
  adjacency[numNodes] = isize(adjacency); // The end offset

  // Sort each span of adjacent indices. This may have a performance benefit, but
  // it is not strictly necessary. For now it is nice just to have a well
  // defined order.
  for (size_t i = 0; i < numNodes; ++i) {
    auto rangeBegin = adjacency.begin() + adjacency[i];
    auto rangeEnd = adjacency.begin() + adjacency[i + 1];
    std::sort(rangeBegin, rangeEnd);
  }

  return adjacency;
}

Obb SimplicialMesh::GetObb() const {
  // TODO[T133705743] - Compute a tight fitting Obb, which may be smaller than the Aabb.
  return mochi::GetObb(_aabb);
}

} // namespace mochi
