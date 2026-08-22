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

#include <mochi_core/geometry/triangular_mesh.h>
#include <mochi_core/utils/array_utils.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/graph_utils.h>
#include <mochi_core/utils/matrix_utils.h>
#include <mochi_core/utils/nd_array_utils.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace mochi {

// Find the edges and boundary edges of a triangular mesh
static void FindTriangularMeshEdges(
    Span<int const> connectivity,
    int numNodesPerElement,
    std::vector<Int2>& outEdges,
    std::vector<Int2>& outBoundaryEdges) {
  MOCHI_ASSERT_VERBOSE(
      connectivity.size() % numNodesPerElement == 0,
      "Connectivity array size must be a multiple of numNodesPerElement");
  outEdges.clear();
  outBoundaryEdges.clear();

  // Maps a sorted edge to the number of elements sharing it
  std::map<Int2, int, Int2Less> edgeToElementCount;

  // For each element
  int const numElements = isize(connectivity) / numNodesPerElement;
  for (int e = 0; e < numElements; ++e) {
    int const base = e * numNodesPerElement;

    // For each edge {i, j}
    for (int i = 0; i < numNodesPerElement; ++i) {
      for (int j = i + 1; j < numNodesPerElement; ++j) {
        Int2 edge = {connectivity[base + i], connectivity[base + j]};

        // Sort the edge indices for unique comparison
        if (edge[0] > edge[1]) {
          std::swap(edge[0], edge[1]);
        }
        ++edgeToElementCount[edge];
      }
    }
  }

  // Output a sorted list of all edges
  outEdges.reserve(edgeToElementCount.size());
  int numBoundaryEdges = 0;
  for (auto const& [edge, count] : edgeToElementCount) {
    outEdges.push_back(edge);
    if (count == 1) {
      ++numBoundaryEdges;
    }
  }

  // Output a sorted list of boundary edges
  outBoundaryEdges.reserve(numBoundaryEdges);
  for (auto const& [edge, count] : edgeToElementCount) {
    if (count == 1) {
      outBoundaryEdges.push_back(edge);
    }
  }
}

// Compute the normal direction and area of each triangular element,
// using the given winding order.
static void CalcTriangleNormalsAndArea(
    Span<Real3 const> coordinates,
    Span<Int3 const> connectivity,
    Span<Real3> outElementNormals,
    Span<real> outElementArea) {
  size_t const numElements = connectivity.size();
  for (size_t e = 0; e < numElements; ++e) {
    Int3 const& F = connectivity[e];
    Real3 const& A = coordinates[F[0]];
    Real3 const& B = coordinates[F[1]];
    Real3 const& C = coordinates[F[2]];
    Real3 const edge[2] = {B - A, C - A};
    Real3 const norm = 0.5_r * Cross(edge[0], edge[1]);
    real const area = Norm(norm);

    // By adding the smallest possible floating-point value, we avoid
    // divide-by-zero and the result is unchanged, except for degenerate cases.
    // See mochi::Normalize.
    if (outElementNormals) {
      outElementNormals[e] = norm / (area + std::numeric_limits<real>::min());
    }
    if (outElementArea) {
      outElementArea[e] = area;
    }
  }
}

// Get the overall barycenter for the mesh
static Real3 CalcMeshBarycenter(
    Span<Real3 const> elementBarycenters,
    Span<real const> elementAreas,
    real totalArea) {
  MOCHI_ASSERT_VERBOSE(
      elementBarycenters.size() == elementAreas.size(), "Should be parallel arrays");
  size_t const numElements = elementBarycenters.size();
  if (numElements == 0) {
    return {};
  }
  MOCHI_ASSERT_VERBOSE(
      IsFinite(totalArea) && totalArea > 0_r, "Total area must be positive and finite.")
  Real3 barycenter = {};
  for (int e = 0; e < numElements; ++e) {
    barycenter += elementBarycenters[e] * elementAreas[e];
  }
  barycenter /= totalArea;
  return barycenter;
}

// Verify that all triangle normals point outward by doing a brute force
// ray-cast against the rest of the mesh. If any normals are found to be
// facing inward, then negate their direction and correct the winding order.
void TriangularMesh::EnsureOutwardNormals() {
  Span<Int3> elementConnectivity = Unflatten<Int3>(Span<int>{_connectivity});
  size_t const numElements = elementConnectivity.size();

  DynamicArray<Matrix2x3r> elementBasis;
  DynamicArray<Matrix2x2r> elementInverseMetric;
  DynamicArray<real> elementOffset;
  elementBasis.reserve(numElements);
  elementInverseMetric.reserve(numElements);
  elementOffset.reserve(numElements);

  for (size_t e = 0; e < numElements; ++e) {
    // Get the induced basis from a affine mapping of a reference unit tet
    Real3 const edge[2] = {
        (_coordinates[elementConnectivity[e][1]] - _coordinates[elementConnectivity[e][0]]),
        (_coordinates[elementConnectivity[e][2]] - _coordinates[elementConnectivity[e][0]])};
    elementBasis.emplace_back(edge[0], edge[1]);

    // Get the inverse metric for later checking if a point lies inside a tri
    real const c0 = Dot(edge[0], edge[0]);
    real const c1 = Dot(edge[1], edge[1]);
    real const c2 = Dot(edge[0], edge[1]);
    real const det = c0 * c1 - c2 * c2;
    elementInverseMetric.emplace_back(Real2{c1, -c2} / det, Real2{-c2, c0} / det);

    // Get the offset that alongside the normal defines the plane
    elementOffset.push_back(-Dot(_elementNormals[e], _elementBarycenters[e]));
  }

  // The number of intersections for each element
  DynamicArray<int> intersection(numElements);
  for (size_t i = 0; i < numElements; ++i) {
    for (size_t j = 0; j < numElements; ++j) {
      if (i == j) {
        continue;
      }

      // Check if planes are parallel
      real const normalsinner = Dot(_elementNormals[i], _elementNormals[j]);

      // If (almost) parallel continue
      if (Abs(normalsinner) < 1.0e-9_r) {
        continue;
      }

      // Determine the intersection on the plane of element j
      real const alpha =
          -(elementOffset[j] + Dot(_elementBarycenters[i], _elementNormals[j])) / normalsinner;
      if (alpha < 0_r) {
        continue;
      }

      Real3 const xintersect = _elementBarycenters[i] + _elementNormals[i] * alpha;

      // Determine if the point is inside the triangle
      Real3 const f = xintersect - _coordinates[elementConnectivity[j][0]];
      Real2 const h = {Dot(f, elementBasis[j][0]), Dot(f, elementBasis[j][1])};
      Real2 const paramCrds = {
          Dot(elementInverseMetric[j][0], h), Dot(elementInverseMetric[j][1], h)};
      if ((paramCrds[0] > 0_r) && (paramCrds[1] > 0_r) && ((paramCrds[0] + paramCrds[1]) <= 1_r)) {
        intersection[i] += 1;
      }
    }
  }

  // Any elements that intersected an odd numer of times must have inward
  // facing.
  for (size_t e = 0; e < numElements; ++e) {
    if (intersection[e] % 2 != 0) {
      // Flip the normal to be outward facing
      _elementNormals[e] *= -1_r;

      // For consistency, also fip the winding order
      std::swap(elementConnectivity[e][1], elementConnectivity[e][2]);
    }
  }
}

// Log a warning if the mesh is not closed
void TriangularMesh::CheckMeshFlux() const {
  MOCHI_ASSERT_VERBOSE(
      _elementNormals.size() == _elementMeasure.size(), "Should be parallel arrays");
  size_t const numElements = _elementNormals.size();
  Real3 sum = {};
  for (size_t i = 0; i < numElements; ++i) {
    sum += (_elementNormals[i] * _elementMeasure[i]);
  }
  real const flux = Norm(sum);
  if (flux < 1.0e-5_r) {
    // Closed surface
  } else {
    MOCHI_LOG_WARNING("Triangle mesh has non-zero residual flux %.9g\n", flux);
  }
}

TriangularMesh::TriangularMesh(Span<Real3 const> coordinatesIn, Span<Int3 const> connectivityIn)
    : SimplicialMesh(coordinatesIn, Flatten(connectivityIn), kNodesPerElement) {
  // Edges & boundary edges (stored in the base class)
  FindTriangularMeshEdges(_connectivity, kNodesPerElement, _edges, _boundaryEdges);
  _edgesActiveNodes.reserve(_edges.size());
  for (Int2 e : _edges) {
    _edgesActiveNodes.emplace_back(_allToActiveMap[e[0]], _allToActiveMap[e[1]]);
  }

  // For a TriangularMesh, each element is a face (but not a volume)
  int const numElements = isize(connectivityIn);
  _numFaces = numElements;
  _numNodesPerFace = 3;
  _numVolumes = 0;

  // Boundary nodes (stored in the base class)
  _boundaryNodes = FindBoundaryNodes(_coordinates.size(), _boundaryEdges);

  // Calculate normals and area according to current winding order
  _elementNormals.resize(numElements);
  _elementMeasure.resize(numElements);
  CalcTriangleNormalsAndArea(
      _coordinates, GetElementConnectivity(), {_elementNormals}, {_elementMeasure});

  bool constexpr kEnsureOutwardNormals = false;
  if constexpr (kEnsureOutwardNormals) {
    // In a closed mesh, all normals should point outward. If they don't, then
    // flip the direction of the normal AND flip the winding order of the triangle
    // in the connectivity array.

    // Generate a temp half-edge data structure to check if the mesh is closed
    auto halfEdge = GenerateHalfEdgeStructure();
    bool const isClosed = IsMeshClosed(halfEdge);
    if (isClosed) {
      EnsureOutwardNormals();
    }
  }

  // Calculate total area
  _totalMeasure = 0_r;
  for (real a : _elementMeasure) {
    _totalMeasure += a;
  }

  // Calculate the overall barycenter (stored in the base class)
  _barycenter = CalcMeshBarycenter(_elementBarycenters, _elementMeasure, _totalMeasure);

  bool constexpr kCheckMeshFlux = false;
  if constexpr (kCheckMeshFlux) {
    // Compute the surface flux to test that element areas and normals are correct. An open surface
    // also produces non-zero flux.
    CheckMeshFlux();
  }
}

void TriangularMesh::ComputeNodeCoordinates(
    Span<Real3 const> nodeDisplacements,
    Span<Real3> outNodeCoordinates) const {
  auto nodeCoordinates = GetNodeCoordinates();
  ArrayAdd(Flatten(outNodeCoordinates), Flatten(nodeDisplacements), Flatten(nodeCoordinates));
}

void TriangularMesh::ComputeElementNormals(
    Span<Real3 const> nodeCoordinates,
    Span<Real3> outElementNormals) const {
  MOCHI_ASSERT_VERBOSE(nodeCoordinates.size() == GetNumNodes(), "Invalid coordinates span");
  MOCHI_ASSERT_VERBOSE(outElementNormals.size() == GetNumElements(), "Invalid output normals span");
  CalcTriangleNormalsAndArea(nodeCoordinates, GetElementConnectivity(), outElementNormals, {});
}

void TriangularMesh::ComputeNodeNormals(Span<Real3> outNodeNormals) const {
  ComputeNodeNormals(GetNodeCoordinates(), GetElementNormals(), outNodeNormals);
}

void TriangularMesh::ComputeNodeNormals(
    Span<Real3 const> nodeCoordinates,
    Span<Real3 const> elementNormals,
    Span<Real3> outNodeNormals) const {
  auto faces = this->GetElementConnectivity();
  size_t const N = GetNumNodes();
  size_t const M = GetNumElements();

  MOCHI_ASSERT_VERBOSE(nodeCoordinates.size() == N, "Invalid coordinates span size");
  MOCHI_ASSERT_VERBOSE(elementNormals.size() == M, "Invalid element normals span size");
  MOCHI_ASSERT_VERBOSE(outNodeNormals.size() == N, "Invalid normals span size");

  // Initialize vertex normals
  for (int i = 0; i < N; ++i) {
    outNodeNormals[i] = Real3{};
  }

  // Compute (angle-weighted) normal
  for (int i = 0; i < M; ++i) {
    Real3 const& normal = elementNormals[i];

    Int3 const& face = faces[i];
    Real3 const& A = nodeCoordinates[face[0]];
    Real3 const& B = nodeCoordinates[face[1]];
    Real3 const& C = nodeCoordinates[face[2]];
    Real3 AB = B - A;
    Real3 AC = C - A;
    Real3 BC = C - B;
    AB = Normalize(AB);
    AC = Normalize(AC);
    BC = Normalize(BC);
    real wA = std::acos(Clamp(Dot(AB, AC), -1_r, 1_r));
    real wB = std::acos(Clamp(Dot(BC, -AB), -1_r, 1_r));
    real wC = kPI - wA - wB;
    outNodeNormals[face[0]] += normal * wA;
    outNodeNormals[face[1]] += normal * wB;
    outNodeNormals[face[2]] += normal * wC;
  }

  // Normalize vertex normals
  for (int i = 0; i < N; ++i) {
    outNodeNormals[i] = Normalize(outNodeNormals[i]);
  }
}

void TriangularMesh::ComputeEdgeNormals(
    HalfEdgeStructure const& halfEdge,
    Span<Real3> outEdgeNormals) const {
  ComputeEdgeNormals(halfEdge, GetElementNormals(), outEdgeNormals);
}

void TriangularMesh::ComputeEdgeNormals(
    HalfEdgeStructure const& halfEdge,
    Span<Real3 const> elementNormals,
    Span<Real3> outEdgeNormals) const {
  size_t const E = this->GetNumEdges();

  MOCHI_ASSERT_VERBOSE(
      elementNormals.size() == this->GetNumFaces(), "Invalid element normals span size");
  MOCHI_ASSERT_VERBOSE(outEdgeNormals.size() == E, "Invalid normals span size");

  // Initialize vertex normals
  for (int i = 0; i < E; ++i) {
    outEdgeNormals[i] = Real3{};
  }

  // Compute per-edge normal
  auto const& halfEdges = halfEdge.halfEdges;
  auto const& edge2halfs = halfEdge.edge2halfs;

  for (int i = 0; i < E; ++i) {
    int heIdx0 = edge2halfs[i][0];
    int heIdx1 = edge2halfs[i][1];

    if (heIdx0 != -1) {
      outEdgeNormals[i] += elementNormals[halfEdges[heIdx0].face];
    }
    if (heIdx1 != -1) {
      outEdgeNormals[i] += elementNormals[halfEdges[heIdx1].face];
    }
  }

  // Normalize edge normals
  for (int i = 0; i < E; ++i) {
    outEdgeNormals[i] = Normalize(outEdgeNormals[i]);
  }
}

std::unordered_map<Int2, std::vector<int>, Int2Hash> GenerateEdgeToElementsMap(
    Span<Int3 const> const& elementConnectivity) {
  std::unordered_map<Int2, std::vector<int>, Int2Hash> edgeToElements;
  int const numElements = isize(elementConnectivity);
  for (int e = 0; e < numElements; ++e) {
    Int3 const& nodeIndices = elementConnectivity[e];
    for (int i = 0; i < 3; ++i) {
      int const n0 = nodeIndices[(i + 1) % 3];
      int const n1 = nodeIndices[(i + 2) % 3];
      Int2 const edge{std::min(n0, n1), std::max(n0, n1)};
      edgeToElements[edge].push_back(e);
    }
  }
  return edgeToElements;
}

std::pair<Graph<int, int>, Graph<int, int>> TriangularMesh::GenerateBendingConnectivityAndStencil()
    const {
  // NOTE: For oriented meshes, this could be generated more simply from a half-edge structure
  // (or the half-edge structure could be used directly), but the following construction is designed
  // to be more general, since edge-opposite nodes are still well-defined for non-orientable meshes
  // (e.g., a triangulation of a Mobius strip).

  // First, generate a mapping from edges to the elements that share each edge.
  Span<Int3 const> const& elementConnectivity = GetElementConnectivity();
  std::unordered_map<Int2, std::vector<int>, Int2Hash> edgeToElements =
      GenerateEdgeToElementsMap(elementConnectivity);

  // Next, iterate the edges of each element and find the opposite node for each edge by using the
  // map generated above.  The first three elements of the stencil connectivity match the base mesh
  // connectivity, and up to three more may be added for edge-opposite nodes.
  int const numElements = GetNumFaces();
  int constexpr kNumStencilNodes = 6;
  int const maxConnectivitySize = kNumStencilNodes * numElements;
  GraphBuilder<int, int> connectivityBuilder(numElements, maxConnectivitySize);
  GraphBuilder<int, int> stencilBuilder(numElements, maxConnectivitySize);
  for (int e = 0; e < numElements; ++e) {
    DynamicArray<int> connectivityRow;
    connectivityRow.reserve(kNumStencilNodes);
    DynamicArray<int> stencilRow;
    stencilRow.reserve(kNumStencilNodes);
    Int3 const& nodeIndices = elementConnectivity[e];
    for (int n = 0; n < 3; n++) {
      int const nodeIndex = nodeIndices[n];
      connectivityRow.push_back(nodeIndex);
      stencilRow.push_back(n);
    }
    for (int edgeIndex = 0; edgeIndex < 3; ++edgeIndex) {
      int const n0 = nodeIndices[(edgeIndex + 1) % 3];
      int const n1 = nodeIndices[(edgeIndex + 2) % 3];
      Int2 const edge{std::min(n0, n1), std::max(n0, n1)};
      MOCHI_ASSERT_VERBOSE(
          edgeToElements.find(edge) != edgeToElements.end(),
          "Found edge not included in edge-to-element map");
      std::vector<int> const& neighbors = edgeToElements[edge];
      for (int const& neighbor : neighbors) {
        // NOTE: In a "honeycomb" type surface, it's possible for an edge to have more than two
        // neighboring elements.  In that case, the edge-opposite node is arbitrarily taken from the
        // first neighbor.
        if (neighbor != e) {
          for (int const& neighborNodeIndex : elementConnectivity[neighbor]) {
            if (neighborNodeIndex != edge[0] && neighborNodeIndex != edge[1]) {
              connectivityRow.push_back(neighborNodeIndex);
              stencilRow.push_back(3 + edgeIndex);
              break;
            }
          }
          break;
        }
      }
    }
    connectivityBuilder.append(connectivityRow);
    stencilBuilder.append(stencilRow);
  }
  return {connectivityBuilder.Build(), stencilBuilder.Build()};
}

HalfEdgeStructure TriangularMesh::GenerateHalfEdgeStructure() const {
  auto faces = GetElementConnectivity();
  auto nodes = GetNodeCoordinates();
  int M = int(faces.size());
  int N = int(nodes.size());
  int E = GetNumEdges();

  // Initialize containers
  std::vector<int> face2half(M, -1);
  std::vector<int> node2half(N, -1);
  std::vector<Int2> edge2halfs;
  edge2halfs.reserve(E);
  std::vector<HalfEdge> halfEdges;
  halfEdges.reserve(3 * M);

  // Create all half-edges
  int countHalf = 0;

  // Map from node indices to half-edge index
  std::map<std::pair<int, int>, int> halfsMap;

  for (int i = 0; i < M; ++i) {
    Int3 const& face = faces[i];

    // Traverse face edges
    for (int e = 0; e < 3; ++e) {
      int node0 = face[(e + 0) % 3];
      int node1 = face[(e + 1) % 3];
      std::pair pair01(node0, node1);

      HalfEdge halfEdge;
      halfEdge.face = i;
      halfEdge.node = node1;
      halfsMap[pair01] = countHalf++;
      halfEdges.push_back(halfEdge);
    }
  }

  // Create structure links

  int countEdge = 0;

  // Map from node indices to edge index
  std::map<std::pair<int, int>, int> edgesMap;

  for (int i = 0; i < M; ++i) {
    Int3 const& face = faces[i];

    // Traverse face edges
    for (int e = 0; e < 3; ++e) {
      int node0 = face[(e + 0) % 3];
      int node1 = face[(e + 1) % 3];
      int node2 = face[(e + 2) % 3];

      // Current and next face half-edge
      int he01 = halfsMap[std::pair(node0, node1)];
      int he12 = halfsMap[std::pair(node1, node2)];

      // Get half-edge pair (if it exists)
      int he10 = -1;
      auto he10Ptr = halfsMap.find(std::pair(node1, node0));
      if (he10Ptr != halfsMap.end()) {
        he10 = he10Ptr->second;
        halfEdges[he01].pair = he10;
      }

      // Create half-edge circle links
      halfEdges[he01].next = he12;

      // Create face -> half-edge links
      if (face2half[i] == -1) {
        face2half[i] = he01;
      }

      // Create node <-> half-edge links
      if (node2half[node0] == -1) {
        node2half[node0] = he01;
      }

      // Create edge <-> half-edge links
      auto sortedPair = std::pair(std::min(node0, node1), std::max(node0, node1));
      if (edgesMap.find(sortedPair) == edgesMap.end()) {
        int edge = countEdge++;
        edgesMap[sortedPair] = edge;

        halfEdges[he01].edge = edge;
        if (he10 != -1) {
          halfEdges[he10].edge = edge;
        }

        edge2halfs.emplace_back(he01, he10);
      }
    }
  }

  // Create final data structure
  return {std::move(halfEdges), std::move(edge2halfs), std::move(node2half), std::move(face2half)};
}

/*********************************************************************************
  Utility Functions
*/

bool IsMeshClosed(HalfEdgeStructure const& halfEdge) {
  for (auto const& he : halfEdge.halfEdges) {
    if (he.pair == -1) {
      return false;
    }
  }
  return true;
}

bool IsMeshClosed(std::unordered_map<Int2, std::vector<int>, Int2Hash> const& edgeToElements) {
  for (auto const& [edge, elements] : edgeToElements) {
    if (elements.size() < 2) {
      return false;
    }
  }
  return true;
}

namespace {
// Custom hash and equality function for Real3: L-infinity norm of a - b < eps
struct Real3Comp {
  explicit Real3Comp(real eps) : oneOverEps(1_r / eps) {}
  real oneOverEps = 1_r / 1e-6_r;
  // Scale each component by 1/eps and convert to int
  inline auto Quantize(Real3 val) const {
    return StaticCast<NdArray<int64_t, 3>>(oneOverEps * val);
  }
  // Hash function
  inline std::size_t operator()(Real3 key) const {
    auto hash3 = Quantize(key);
    return hash3[0] ^ (hash3[1] << 1) ^ (hash3[2] << 2);
  }
  // Equality function
  inline bool operator()(Real3 a, Real3 b) const {
    return Quantize(a) == Quantize(b);
  }
};
} // namespace

std::shared_ptr<TriangularMesh const> TriangularMeshFromClusteredVertices(
    Span<Real3 const> coordinates,
    Span<Int3 const> connectivity,
    real distThreshold) {
  // Map from old indices to new indices
  std::vector<int> indexMap(coordinates.size());

  // Map from vertex locations to new indices
  Real3Comp comp(distThreshold);
  std::unordered_map<Real3, int, Real3Comp, Real3Comp> posMap(0, comp, comp);

  // Container of new coordinates
  std::vector<Real3> coordinatesNew;
  coordinatesNew.reserve(coordinates.size());

  // Add vertices to posMap, indexMap and coordinatesNew
  for (int iOld = 0; iOld < coordinates.size(); iOld++) {
    if (auto iNew = posMap.find(coordinates[iOld]); iNew != posMap.end()) {
      indexMap[iOld] = iNew->second;
    } else {
      indexMap[iOld] = static_cast<int>(coordinatesNew.size());
      posMap.insert({coordinates[iOld], indexMap[iOld]});
      coordinatesNew.emplace_back(coordinates[iOld]);
    }
  }

  // Return null if no vertices were clustered
  if (coordinatesNew.size() == coordinates.size()) {
    return {};
  }

  // Redefine connectivity and check for degenerate triangles
  bool degenerate = false;
  std::vector<Int3> connectivityNew;
  connectivityNew.reserve(connectivity.size());
  for (auto const& tri : connectivity) {
    auto aNew = indexMap[tri[0]];
    auto bNew = indexMap[tri[1]];
    auto cNew = indexMap[tri[2]];
    if (aNew == bNew || aNew == cNew || bNew == cNew) {
      degenerate = true;
      break;
    }
    connectivityNew.emplace_back(aNew, bNew, cNew);
  }

  // Return null if degenerate
  if (degenerate) {
    return {};
  }

  // Return new triangle mesh
  return std::make_shared<TriangularMesh>(coordinatesNew, connectivityNew);
}

std::pair<std::vector<Real3>, std::vector<Int3>>
UniformSquareTriangularMeshData(Int2 n, Real2 scale, int axis) {
  MOCHI_ASSERT(n[0] > 0 && n[1] > 0, "Invalid number of elements");
  MOCHI_ASSERT(axis >= 0 && axis < 3, "Invalid axis");
  int const kVertsI = n[0] + 1;
  int const kVertsJ = n[1] + 1;
  std::vector<Real3> coordinates((size_t)(kVertsI * kVertsJ));
  for (int i = 0; i < kVertsI; ++i) {
    for (int j = 0; j < kVertsJ; ++j) {
      Real3& coords = coordinates[i + j * kVertsI];
      coords[(axis + 1) % 3] = scale[0] * (static_cast<real>(i) / n[0] - 0.5_r);
      coords[(axis + 2) % 3] = scale[1] * (static_cast<real>(j) / n[1] - 0.5_r);
      coords[axis] = 0_r;
    }
  }
  std::vector<Int3> connectivity(2 * (size_t)(n[0] * n[1]));
  for (int i = 0; i < n[0]; ++i) {
    for (int j = 0; j < n[1]; ++j) {
      int const base = 2 * (i + j * n[0]);
      connectivity[base] = Int3{i + j * kVertsI, i + 1 + j * kVertsI, i + (j + 1) * kVertsI};
      connectivity[base + 1] =
          Int3{i + 1 + j * kVertsI, i + 1 + (j + 1) * kVertsI, i + (j + 1) * kVertsI};
    }
  }
  return {coordinates, connectivity};
}

} // namespace mochi
