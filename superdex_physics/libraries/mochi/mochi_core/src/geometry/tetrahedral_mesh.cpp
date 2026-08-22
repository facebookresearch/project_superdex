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
#include <mochi_core/geometry/model_utils.h>
#include <mochi_core/geometry/tetrahedral_mesh.h>
#include <mochi_core/utils/array_utils.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/nd_array_utils.h>

#include <algorithm>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(__APPLE__) // TODO: Move this
#include <strings.h>
#endif

namespace mochi {

// Calculate the volume for each tetrahedral element
static std::vector<real> CalcElementVolume(
    Span<Real3 const> coordinates,
    Span<Int4 const> connectivity) {
  std::vector<real> elementVolume(connectivity.size());
  for (size_t e = 0; e < connectivity.size(); ++e) {
    // Get the induced basis from a affine mapping of a reference unit tet
    Real3 const vert[4] = {
        coordinates[connectivity[e][0]],
        coordinates[connectivity[e][1]],
        coordinates[connectivity[e][2]],
        coordinates[connectivity[e][3]]};
    Real3 const g[3] = {vert[1] - vert[0], vert[2] - vert[0], vert[3] - vert[0]};

    // Compute the volume
    elementVolume[e] = Dot(g[2], Cross(g[0], g[1])) / 6_r;
  }
  return elementVolume;
}

namespace impl {

struct ElementInfo {
  size_t count = 0; // Number of touching elements
  size_t element = 0; // Index of a touching element
  size_t faceNum = 0; // the index of kTetFaceIndices
};

static std::unordered_map<Int3, ElementInfo, Int3SortAndHash, Int3SortAndCompare>
CreateFaceToElementMap(Span<Int4 const> connectivity) {
  // Find the elements that share each unique face
  std::unordered_map<Int3, ElementInfo, Int3SortAndHash, Int3SortAndCompare> faceToElement;
  for (size_t e = 0; e < connectivity.size(); ++e) {
    for (size_t f = 0; f < std::size(TetFaces::kIndices); ++f) {
      auto const& combo = TetFaces::kIndices[f];
      Int3 faceOrder = {
          connectivity[e][combo[0]], connectivity[e][combo[1]], connectivity[e][combo[2]]};

      // Add or update map entry
      ElementInfo& info = faceToElement[faceOrder];
      ++info.count;
      info.element = e;
      info.faceNum = f;
    }
  }
  return faceToElement;
}

} // namespace impl

// Get the unique triangular faces on the surface of a TetrahedralMesh
static std::vector<Int3> FindBoundaryFaces(
    Span<Real3 const> coordinates,
    Span<Int4 const> connectivity,
    Span<Real3 const> elementBarycenters,
    int* outNumUniqueFaces,
    std::vector<TetrahedralMesh::BoundaryFaceInfo>* outBoundaryFaces = nullptr) {
  auto const faceToElement = impl::CreateFaceToElementMap(connectivity);

  // Return the number of unique faces (including interior faces)
  *outNumUniqueFaces = isize(faceToElement);

  // Count the boundary faces (faces referenced by exactly one element)
  size_t numBoundaryFaces = 0;
  for (auto const& [face, info] : faceToElement) {
    if (info.count == 1) {
      ++numBoundaryFaces;
    }
  }

  // Build a flat list of boundary faces. Fix the winding order as needed.
  std::vector<Int3> boundaryFacesConnectivity;
  boundaryFacesConnectivity.reserve(numBoundaryFaces);
  if (outBoundaryFaces) {
    outBoundaryFaces->reserve(numBoundaryFaces);
  }
  for (auto const& [face, info] : faceToElement) {
    if (info.count == 1) {
      // Compute the direction of the normal implied by the current winding
      // order
      Real3 const edge[2] = {
          coordinates[face[1]] - coordinates[face[0]], coordinates[face[2]] - coordinates[face[0]]};
      Real3 const norm = Cross(edge[0], edge[1]);

      // The normal should point away from the center of the tetrahedron.
      MOCHI_ASSERT(
          Dot(norm, coordinates[face[0]] - elementBarycenters[info.element]) > 0_r,
          "Tet face normal points inward");

      boundaryFacesConnectivity.push_back(face);
      if (outBoundaryFaces) {
        outBoundaryFaces->push_back(
            TetrahedralMesh::BoundaryFaceInfo{(int)info.element, (int)info.faceNum});
      }
    }
  }

  return boundaryFacesConnectivity;
}

static std::vector<Int2> FindAllEdges(Span<int const> connectivity, int numNodesPerElement) {
  std::unordered_set<Int2, Int2Hash> uniqueEdges;
  int const numElements = isize(connectivity) / numNodesPerElement;
  for (int e = 0; e < numElements; ++e) {
    int const base = e * numNodesPerElement;
    for (int i = 0; i < numNodesPerElement; ++i) {
      for (int j = i + 1; j < numNodesPerElement; ++j) {
        Int2 edge = {connectivity[base + i], connectivity[base + j]};
        // Sort the indices in the edge for unique comparison
        if (edge[0] > edge[1]) {
          std::swap(edge[0], edge[1]);
        }
        uniqueEdges.insert(edge);
      }
    }
  }
  // Nice to have a well defined order, but not strictly necessary
  std::vector<Int2> uniqueEdgeList(uniqueEdges.begin(), uniqueEdges.end());
  std::sort(uniqueEdgeList.begin(), uniqueEdgeList.end(), Int2Less{});
  return uniqueEdgeList;
}

static std::vector<int> BuildBoundaryFaceAdjacency(size_t numNodes, Span<Int3 const> faces) {
  // For each node, build a set of unique faces that share it.
  std::vector<std::unordered_set<int>> adjacencySets;
  adjacencySets.resize(numNodes);
  for (int iFace = 0; iFace < isize(faces); ++iFace) {
    for (int iNode : faces[iFace]) {
      adjacencySets[iNode].insert(iFace);
    }
  }

  // Calculate the space needed
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

static std::vector<int> BuildNodeAdjacency(size_t numNodes, Span<Int4 const> connectivity) {
  // For each node, build a set of unique nodes adjacent to it
  std::vector<std::unordered_set<int>> nodeAdjacencySets;
  nodeAdjacencySets.resize(numNodes);
  for (Int4 const& elem : connectivity) {
    for (int i : elem) {
      for (int j : elem) {
        if (j != i) {
          nodeAdjacencySets[i].insert(j);
        }
      }
    }
  }

  // Calculate the space needed
  size_t nodeAdjacencyDataSize = numNodes + 1; // +1 to store the end offset
  for (auto const& set : nodeAdjacencySets) {
    nodeAdjacencyDataSize += set.size();
  }

  // nodeAdjacency[i] is the offset of the start of the adjacency data for node
  // i. nodeAdjacency[i+1] - nodeAdjacency[i] is the number of adjacent nodes.
  std::vector<int> nodeAdjacency;
  nodeAdjacency.reserve(nodeAdjacencyDataSize);
  nodeAdjacency.resize(numNodes + 1);
  for (size_t i = 0; i < numNodes; ++i) {
    auto const& adjacentNodes = nodeAdjacencySets[i];
    nodeAdjacency[i] = isize(nodeAdjacency);
    nodeAdjacency.insert(nodeAdjacency.end(), adjacentNodes.begin(), adjacentNodes.end());
  }
  MOCHI_ASSERT_VERBOSE(
      nodeAdjacency.size() == nodeAdjacencyDataSize, "Adjacency table offsets are incorrect");
  nodeAdjacency[numNodes] = isize(nodeAdjacency); // The end offset

  // Sort each span of adjacent nodes. This may have a performance benefit, but
  // it is not strictly necessary. For now it is nice just to have a well
  // defined order.
  for (size_t i = 0; i < numNodes; ++i) {
    auto rangeBegin = nodeAdjacency.begin() + nodeAdjacency[i];
    auto rangeEnd = nodeAdjacency.begin() + nodeAdjacency[i + 1];
    std::sort(rangeBegin, rangeEnd);
  }

  return nodeAdjacency;
}

std::unique_ptr<TetrahedralMesh> LoadTetrahedralMesh(std::string const& filename, Error& error) {
  MOCHI_ERROR_RETURN(error, {});

  // Load the model file
  ModelData model = model::LoadFromFile(filename, error);

  // The model must contain a tetrahedral mesh
  MOCHI_ERROR_IF(
      !model.mesh.has_value() || model.mesh->nodesPerElement != 4,
      error,
      "File does not contain a tetrahedral mesh");
  MOCHI_ERROR_RETURN(error, {});

  // TODO: Pass ownership of the memory rather than copying
  return std::make_unique<TetrahedralMesh>(
      Unflatten<Real3 const>(MakeConstSpan(model.mesh->coordinates)),
      Unflatten<Int4 const>(MakeConstSpan(model.mesh->connectivity)));
}

TriangularMesh CreateBoundaryMesh(TetrahedralMesh const& mesh) {
  auto nodes = mesh.GetNodeCoordinates();
  auto faces = mesh.GetBoundaryFacesConnectivity();

  // Collect boundary node indices in insertion order (first appearance).
  std::unordered_set<int> seenNodes;
  std::vector<int> boundaryNodeList;
  for (auto face : faces) {
    for (size_t n = 0; n < 3; ++n) {
      if (seenNodes.insert(face[n]).second) {
        boundaryNodeList.push_back(face[n]);
      }
    }
  }

  // Build vol->boundary mapping
  std::vector<int> mapTetNode2TriNodeIdx(mesh.GetNumNodes(), -1);
  std::vector<Real3> newNodes;
  newNodes.reserve(boundaryNodeList.size());
  for (int volIdx : boundaryNodeList) {
    mapTetNode2TriNodeIdx[volIdx] = int(newNodes.size());
    newNodes.push_back(nodes[volIdx]);
  }

  // Remap connectivity
  std::vector<Int3> newFaces(faces.size());
  for (size_t f = 0; f < faces.size(); ++f) {
    for (size_t n = 0; n < 3; ++n) {
      newFaces[f][n] = mapTetNode2TriNodeIdx[faces[f][n]];
    }
  }

  return {newNodes, newFaces};
}

/*********************************************************************************
  TetrahedralMesh Class
*/
TetrahedralMesh::TetrahedralMesh(
    Span<Real3 const> const& coordinatesIn,
    Span<Int4 const> const& connectivityIn,
    std::vector<BoundaryFaceInfo>* boundaryFaces,
    std::vector<Int3>* boundaryFacesConnectivity)
    : SimplicialMesh(coordinatesIn, Flatten(connectivityIn), kNodesPerElement) {
  MOCHI_PROFILE_SCOPE();

  // ensure the optional args are either both usable or both nullptr
  // because it would not make sense to pass only one of them
  MOCHI_ASSERT(
      (boundaryFaces && boundaryFacesConnectivity) ||
      (!boundaryFaces && !boundaryFacesConnectivity));

  Span<Int4 const> connectivity = GetElementConnectivity();
  size_t const numElements = connectivity.size();
  _numNodesPerFace = 3;

  // For each node, find all adjacent nodes. Store it all in _nodeAdjacency
  _nodeAdjacency = BuildNodeAdjacency(_coordinates.size(), connectivity);

  // For a TetrahedralMesh, each element is one "volume"
  _numVolumes = (int)numElements;

  // Calculate the volume of each element
  _elementMeasure = CalcElementVolume(_coordinates, connectivity);

  // If any nodes had negative volume, then fix the node ordering
  Span<Int4> connectivityMutable = Unflatten<Int4>(Span<int>{_connectivity});
  for (size_t e = 0; e < numElements; ++e) {
    if (_elementMeasure[e] < 0_r) {
      std::swap(connectivityMutable[e][0], connectivityMutable[e][1]);
      _elementMeasure[e] = -_elementMeasure[e];
    }
  }

  // compute total volume
  _totalMeasure = 0_r;
  for (real v : _elementMeasure) {
    _totalMeasure += v;
  }

  // handle boundaries (faces and edges)
  if (!boundaryFaces && !boundaryFacesConnectivity) {
    _boundaryFacesConnectivity = FindBoundaryFaces(
        _coordinates, connectivity, _elementBarycenters, &_numFaces, &_boundaryFaces);
  } else {
    /*a single else is sufficient here because the assert at the beginning of the method
    guarantees this else can only be the case when they are both valid pointers because of the */
    _boundaryFaces = *boundaryFaces;
    _boundaryFacesConnectivity = *boundaryFacesConnectivity;

    /*
        PHIL: This was previously
        _numFaces = impl::CountUniqueFaces(connectivity);

        However, hyper-reduction tetrahedral meshes might mean that not all possible faces on the
        boundary are actually in the mesh, if that is the case we shouldn't count them.
    */
    _numFaces = isize(_boundaryFacesConnectivity);
  }
  FindAllBoundaryInformation();

  _boundaryMesh = std::make_shared<TriangularMesh>(_coordinates, _boundaryFacesConnectivity);

  // Calculate the mesh barycenter
  // (average of element barycenters weighted by volume)
  _barycenter = {};
  if (_numVolumes > 0) {
    for (size_t e = 0; e < _numVolumes; ++e) {
      _barycenter += _elementBarycenters[e] * _elementMeasure[e];
    }
    _barycenter /= _totalMeasure;
  }
}

void TetrahedralMesh::FindAllBoundaryInformation() {
  _numBoundaryFaces = _boundaryFacesConnectivity.size();

  // For each node, find all boundary faces that share it. Store it all in _boundaryFaceAdjacency
  _boundaryFaceAdjacency =
      BuildBoundaryFaceAdjacency(_coordinates.size(), _boundaryFacesConnectivity);

  // Edges (stored in the base class)
  _edges = FindAllEdges(GetFlatConnectivity(), kNodesPerElement);

  // Edges in active-node index space (stored in the base class)
  MOCHI_ASSERT(_edgesActiveNodes.empty(), "Active-node edge data should only be initialized once.");
  _edgesActiveNodes.reserve(_edges.size());
  for (Int2 const& edge : _edges) {
    _edgesActiveNodes.emplace_back(_allToActiveMap[edge[0]], _allToActiveMap[edge[1]]);
  }

  // Boundary edges (stored in the base class)
  _boundaryEdges = FindAllEdges(Flatten(Span<Int3 const>{_boundaryFacesConnectivity}), 3);

  // Boundary nodes (stored in the base class)
  _boundaryNodes = FindBoundaryNodes(_coordinates.size(), _boundaryEdges);
}

int TetrahedralMesh::GetElementAt(Real3 const& query) const {
  int elementCount = GetNumElements();
  for (int element = 0; element < elementCount; ++element) {
    int connectivityIdx = element * kNodesPerElement;
    int node1 = _connectivity[connectivityIdx];
    int node2 = _connectivity[connectivityIdx + 1];
    int node3 = _connectivity[connectivityIdx + 2];
    int node4 = _connectivity[connectivityIdx + 3];

    if (IsInsideTetrahedron(
            _coordinates[node1],
            _coordinates[node2],
            _coordinates[node3],
            _coordinates[node4],
            query)) {
      return element;
    }
  }

  return -1;
}

bool TetrahedralMesh::GetRigidPivot(int& outElementIndex, Real3& outPosition) const {
  int const numElements = GetNumElements();
  if (!numElements) {
    return false;
  }

  // By default, we place the pivot at the center-of-mass = the barycenter under uniform density
  outPosition = _barycenter;
  outElementIndex = GetElementAt(_barycenter);

  // But if the center-of-mass is outside the mesh, then we have to pick a different point. Ideally
  // it would be somewhere that doesn't move much except when the whole actor is moving. In the
  // future, we might want to let the user (or offline tool) select this point. For now, lets use
  // the center of the nearest element.
  if (outElementIndex < 0) {
    Vec4r com = ToSimd(_barycenter);
    Vec4r closestPt = SimdZero();
    real closestDistSqr = std::numeric_limits<float>::infinity();
    int closestElement = -1;
    for (int i = 0; i < numElements; ++i) {
      Vec4r pt = ToSimd(GetElementBarycenter(i));
      Vec4r delta = pt - com;
      real distSqr = Dot<3>(delta, delta);
      if (distSqr < closestDistSqr) {
        closestPt = pt;
        closestDistSqr = distSqr;
        closestElement = i;
      }
    }
    MOCHI_ASSERT(closestElement >= 0);
    outPosition = ToReal3(closestPt);
    outElementIndex = closestElement;
  }

  return outElementIndex >= 0;
}

std::pair<std::vector<Real3>, std::vector<Int4>> UniformCubeTetMeshData(Int3 n, Real3 scale) {
  MOCHI_ASSERT(n[0] > 0 && n[1] > 0 && n[2] > 0, "Invalid number of elements");
  int const numNodeX = n[0] + 1;
  int const numNodeY = n[1] + 1;
  int const numNodeZ = n[2] + 1;
  int const numNodes = numNodeX * numNodeY * numNodeZ;
  Real3 const cubeScale = scale / StaticCast<Real3>(n);

  // Create coordinates in X, Y, Z order
  std::vector<Real3> coordinates;
  coordinates.reserve(numNodes);
  for (int k = 0; k < numNodeZ; ++k) {
    for (int j = 0; j < numNodeY; ++j) {
      for (int i = 0; i < numNodeX; ++i) {
        coordinates.emplace_back(i * cubeScale[0], j * cubeScale[1], k * cubeScale[2]);
      }
    }
  }

  // Create connectivity
  std::vector<Int4> connectivity;
  int const lineSize = numNodeX;
  int const slideSize = numNodeX * numNodeY;
  connectivity.reserve(n[0] * n[1] * n[2] * 5);
  for (int k = 0; k < n[2]; ++k) {
    for (int j = 0; j < n[1]; ++j) {
      for (int i = 0; i < n[0]; ++i) {
        int offset = slideSize * k + lineSize * j + i;

        int const v[8] = {
            offset,
            offset + 1,
            offset + lineSize,
            offset + lineSize + 1,
            offset + slideSize,
            offset + slideSize + 1,
            offset + slideSize + lineSize,
            offset + slideSize + lineSize + 1};

        //         2 ------- 3
        //       / |       / |
        //      /  |      /  |
        //     6 ------- 7   |
        //     |   0 ----|-- 1
        //     |  /      |  /
        //     | /       | /
        //     4 ------- 5

        // 5-tet hex decomposition: alternate the central tet diagonal in a 3D checkerboard so that
        // adjacent hexes produce matching face triangulations.
        if ((i + j + k) % 2 == 0) {
          connectivity.emplace_back(v[0], v[1], v[2], v[4]); // corner 0
          connectivity.emplace_back(v[6], v[7], v[4], v[2]); // corner 6
          connectivity.emplace_back(v[5], v[4], v[7], v[1]); // corner 5
          connectivity.emplace_back(v[3], v[2], v[1], v[7]); // corner 3
          connectivity.emplace_back(v[1], v[2], v[4], v[7]); // central
        } else {
          connectivity.emplace_back(v[1], v[0], v[5], v[3]); // corner 1
          connectivity.emplace_back(v[2], v[0], v[3], v[6]); // corner 2
          connectivity.emplace_back(v[4], v[0], v[6], v[5]); // corner 4
          connectivity.emplace_back(v[7], v[3], v[5], v[6]); // corner 7
          connectivity.emplace_back(v[0], v[3], v[6], v[5]); // central
        }
      }
    }
  }

  return {coordinates, connectivity};
}

} // namespace mochi
