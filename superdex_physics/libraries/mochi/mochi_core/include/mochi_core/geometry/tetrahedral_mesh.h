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
#include <mochi_core/geometry/triangular_mesh.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/error.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/nd_array_utils.h>
#include <mochi_core/utils/span.h>
#include <mochi_core/utils/transform_srt.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace mochi {

/*********************************************************************************
  Tetrahedral Mesh Class
*/
class TetrahedralMesh final : public SimplicialMesh {
 public:
  static constexpr int kNodesPerElement = 4;

  // To keep track of boundary elements
  // the face num ranges from 0-3 where the faces
  // are enumerated as follows
  // 0 -> (0,1,2)
  // 1 -> (0,1,3)
  // 2 -> (0,2,3)
  // 3 -> (1,2,3)
  struct BoundaryFaceInfo {
    int element;
    int faceNum;
  };

  // Construct a TetrahedralMesh and copy the data.
  // The two optional args are useful when the boundary faces are known :
  // this is needed for example when building a tet mesh
  // with disjoint elements and we already know which elements are
  // the "physical" boundary ones.
  TetrahedralMesh(
      Span<Real3 const> const& coordinates,
      Span<Int4 const> const& connectivity,
      std::vector<BoundaryFaceInfo>* boundaryFaces = nullptr,
      std::vector<Int3>* boundaryFacesConnectivity = nullptr);

  // Helper for unit test that synthesis the pair of inputs
  TetrahedralMesh(std::pair<std::vector<Real3>, std::vector<Int4>> const& pair)
      : TetrahedralMesh(MakeSpan(pair.first), MakeSpan(pair.second)) {}

  // Get the node indices for each element
  Span<Int4 const> GetElementConnectivity() const;

  // Get the full volume-mesh node indices for each triangular face that belongs to exactly one
  // tetrahedral element.
  Span<Int3 const> GetBoundaryFacesConnectivity() const;

  // Given the index of a node, return the indices of all boundary faces that share it.
  Span<int const> GetAdjacentBoundaryFaces(int nodeIndex) const;

  // Given the index of a node, return the indices of all other nodes that share and edge with it.
  Span<int const> GetAdjacentNodes(int nodeIndex) const;

  // Returns the number of boundary faces
  size_t GetNumBoundaryFaces() const;

  // Returns a span of BoundaryFaceInfo for each boundary
  // face on the boundary of the domain
  Span<BoundaryFaceInfo const> GetBoundaryFaces() const;

  // Returns the boundary mesh built on the entire set of volume mesh nodes. Its connectivity uses
  // volume-mesh node indices, so the returned TriangularMesh may contain nodes unused by any
  // boundary triangle. Use CreateBoundaryMesh() when a boundary-node-only surface representation is
  // needed.
  std::shared_ptr<TriangularMesh const> const& GetBoundaryMesh() const;

  // Obtain the index of the element which contains the query (query).
  // NOTE: This runs without an acceleration structure and therefore requires
  // O(n) time. Returns -1 if no element contains the query.
  int GetElementAt(Real3 const& query) const;

  // Get a position in the mesh that can be used to evaluate the "rigid pivot", which in turn will
  // let us approximate to global rotation and translation of the soft actor (e.g. when the whole
  // thing is moving). By default, this point will be located at the center-of-mass of the reference
  // mesh, but a different point might have been selected if that would be outside the volume.
  // Return true on success.
  bool GetRigidPivot(int& outElementIndex, Real3& outPosition) const;

 private:
  void FindAllBoundaryInformation();

 private:
  std::vector<BoundaryFaceInfo> _boundaryFaces;
  std::shared_ptr<TriangularMesh const> _boundaryMesh;
  size_t _numBoundaryFaces = 0;
  std::vector<Int3> _boundaryFacesConnectivity;
  std::vector<int> _boundaryFaceAdjacency;
  std::vector<int> _nodeAdjacency;
};

/*********************************************************************************
  Utility Functions
*/

/**
  Load a TetrahedralMesh from a JSON file
  @param filename The path of the JSON file to load
*/
std::unique_ptr<TetrahedralMesh> LoadTetrahedralMesh(std::string const& filename, Error& error);

/**
  Create a boundary-node-only @ref TriangularMesh from the boundary of a @ref TetrahedralMesh.
  Output nodes contain only boundary nodes, in insertion order (first appearance while iterating
  boundary faces), and connectivity is remapped into that node list.

  @param[in] mesh The tetrahedral mesh whose boundary is extracted.
*/
TriangularMesh CreateBoundaryMesh(TetrahedralMesh const& mesh);

/*********************************************************************************
  Inlines
*/

inline Span<Int4 const> TetrahedralMesh::GetElementConnectivity() const {
  return Unflatten<Int4 const>(GetFlatConnectivity());
}

inline Span<Int3 const> TetrahedralMesh::GetBoundaryFacesConnectivity() const {
  return {_boundaryFacesConnectivity};
}

inline Span<int const> TetrahedralMesh::GetAdjacentBoundaryFaces(int nodeIndex) const {
  return mochi::GetAdjacentIndices(nodeIndex, _boundaryFaceAdjacency);
}

inline Span<int const> TetrahedralMesh::GetAdjacentNodes(int nodeIndex) const {
  return mochi::GetAdjacentIndices(nodeIndex, _nodeAdjacency);
}

inline size_t TetrahedralMesh::GetNumBoundaryFaces() const {
  return _numBoundaryFaces;
}

inline Span<TetrahedralMesh::BoundaryFaceInfo const> TetrahedralMesh::GetBoundaryFaces() const {
  return {_boundaryFaces};
}

inline std::shared_ptr<TriangularMesh const> const& TetrahedralMesh::GetBoundaryMesh() const {
  return _boundaryMesh;
}

// This is a utility function intended to be used in samples and unit tests, generating the nodes
// and connectivity corresponding to a uniform tetrahedral mesh of a(n optionally scaled) unit cube.
// It is defined outside of the respective sample and test utility/helper headers to be accessible
// in both places. It produces a mesh of n[0] x n[1] x n[2] hexahedral cells, each decomposed into 5
// tetrahedra using a 3D checkerboard alternation pattern (Freudenthal) that ensures face-conforming
// meshes between adjacent cells.
std::pair<std::vector<Real3>, std::vector<Int4>> UniformCubeTetMeshData(
    Int3 n,
    Real3 scale = Real3{1_r, 1_r, 1_r});

} // namespace mochi
