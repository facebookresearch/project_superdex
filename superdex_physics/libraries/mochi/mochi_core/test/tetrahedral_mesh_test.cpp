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
#include <mochi_core/geometry/tetrahedral_mesh.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/matrix_utils.h>
#include <mochi_core/utils/nd_array_utils.h>

#include <algorithm>
#include <array>
#include <vector>

using namespace mochi;

TEST(TetrahedralMesh, Cube) {
  // A solid unit cube with one corner at (0,0,0)
  //
  //         6 ------- 7
  //       / |       / |
  //      /  |      /  |
  //     2 ------- 3   |
  //     |   4 ----|-- 5
  //     |  /      |  /
  //     | /       | /
  //     0 ------- 1
  //
  std::vector<Real3> const coordinates = {
      Real3{0.0_r, 0.0_r, 0.0_r}, // 0
      Real3{1.0_r, 0.0_r, 0.0_r}, // 1
      Real3{0.0_r, 1.0_r, 0.0_r}, // 2
      Real3{1.0_r, 1.0_r, 0.0_r}, // 3
      Real3{0.0_r, 0.0_r, 1.0_r}, // 4
      Real3{1.0_r, 0.0_r, 1.0_r}, // 5
      Real3{0.0_r, 1.0_r, 1.0_r}, // 6
      Real3{1.0_r, 1.0_r, 1.0_r}, // 7
  };
  std::vector<Int4> const connectivity = {
      Int4{2, 6, 3, 0}, // corner vert 2
      Int4{7, 3, 6, 5}, // corner vert 7
      Int4{1, 3, 5, 0}, // corner vert 1
      Int4{4, 0, 5, 6}, // corner vert 4
      Int4{6, 0, 3, 5}, // the one fully interior tetrahedron
  };

  TetrahedralMesh mesh(coordinates, connectivity);
  static_assert(3 == mesh.kSpaceDimension);
  static_assert(4 == mesh.kNodesPerElement);

  EXPECT_EQ(8, mesh.GetNumNodes());
  EXPECT_EQ(18, mesh.GetNumEdges());
  EXPECT_EQ(16, mesh.GetNumFaces());
  EXPECT_EQ(5, mesh.GetNumVolumes()); // Same as GetNumElements() for a TetrahedralMesh
  EXPECT_EQ(5, mesh.GetNumElements());
  EXPECT_EQ(4, mesh.GetNumNodesPerElement());

  // Node coordinates
  auto nodeCoords = mesh.GetNodeCoordinates();
  EXPECT_SPAN_EQ(coordinates, nodeCoords);

  // Expect all 5 tets but expect the indicies within each tet to be
  // reordered to match the convention that: cross(v1-v0, v2-v0) should point toward v3
  std::vector<Int4> const expectedConnectivity = {
      Int4{6, 2, 3, 0}, // reordered
      Int4{3, 7, 6, 5}, // reordered
      Int4{3, 1, 5, 0}, // reordered
      Int4{0, 4, 5, 6}, // reordered
      Int4{6, 0, 3, 5}, // not reordered
  };
  EXPECT_SPAN_EQ(expectedConnectivity, mesh.GetElementConnectivity());

  // Connectivity as a flat list
  EXPECT_SPAN_EQ(Flatten(Span<Int4 const>{expectedConnectivity}), mesh.GetFlatConnectivity());

  //
  // Nodes neighboring a subset of elements
  //
  std::vector<int> nodesInElems;

  // No elements
  {
    mesh.UniqueNodesInElements(std::vector<int>{}, nodesInElems);
    EXPECT_TRUE(nodesInElems.empty());
  }

  // Sorted, unique subset of elements
  {
    mesh.UniqueNodesInElements(std::vector<int>{0, 3, 4}, nodesInElems);
    std::sort(nodesInElems.begin(), nodesInElems.end());
    EXPECT_SPAN_EQ(MakeConstSpan(std::vector<int>{0, 2, 3, 4, 5, 6}), nodesInElems);
  }

  // Unsorted, unique subset of elements
  {
    mesh.UniqueNodesInElements(std::vector<int>{4, 0, 2}, nodesInElems);
    std::sort(nodesInElems.begin(), nodesInElems.end());
    EXPECT_SPAN_EQ(MakeConstSpan(std::vector<int>{0, 1, 2, 3, 5, 6}), nodesInElems);
  }

  // Unsorted, non-unique subset of elements
  {
    mesh.UniqueNodesInElements(std::vector<int>{4, 0, 4, 2, 4, 2}, nodesInElems);
    std::sort(nodesInElems.begin(), nodesInElems.end());
    EXPECT_SPAN_EQ(MakeConstSpan(std::vector<int>{0, 1, 2, 3, 5, 6}), nodesInElems);
  }

  // All elements
  {
    mesh.UniqueNodesInElements(std::vector<int>{0, 1, 2, 3, 4}, nodesInElems);
    std::sort(nodesInElems.begin(), nodesInElems.end());
    EXPECT_SPAN_EQ(MakeConstSpan(std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7}), nodesInElems);
  }

  // Boundary nodes (all nodes are boundary nodes)
  std::vector<int> const expectedBoundaryNodes = {0, 1, 2, 3, 4, 5, 6, 7};
  EXPECT_TRUE(test::EqualSpanUnordered(expectedBoundaryNodes, mesh.GetBoundaryNodes()));

  // Edges (any order of the correct pairs will be accepted here)
  std::vector<Int2> expectedEdges = {
      Int2{0, 1},
      Int2{1, 5},
      Int2{4, 5},
      Int2{0, 4},
      Int2{2, 3},
      Int2{3, 7},
      Int2{6, 7},
      Int2{2, 6},
      Int2{0, 2},
      Int2{1, 3},
      Int2{5, 7},
      Int2{4, 6},
      Int2{0, 3},
      Int2{0, 5},
      Int2{0, 6},
      Int2{3, 5},
      Int2{3, 6},
      Int2{5, 6},
  };
  std::vector<Int2> actualEdges(mesh.GetEdges().begin(), mesh.GetEdges().end());
  for (Int2 edge : actualEdges) {
    std::sort(edge.begin(), edge.end());
  }

  auto int2_less = [](Int2 const& a, Int2 const& b) {
    return (a[0] < b[0]) || (a[0] == b[0] && a[1] < b[1]);
  };
  std::sort(actualEdges.begin(), actualEdges.end(), int2_less);
  std::sort(expectedEdges.begin(), expectedEdges.end(), int2_less);
  EXPECT_SPAN_EQ(expectedEdges, actualEdges);

  // Boundary edges (all edges are boundary edges in this mesh)
  std::vector<Int2> actualBoundaryEdges(mesh.GetEdges().begin(), mesh.GetEdges().end());
  for (Int2 edge : actualBoundaryEdges) {
    std::sort(edge.begin(), edge.end());
  }
  std::sort(actualBoundaryEdges.begin(), actualBoundaryEdges.end(), int2_less);
  EXPECT_SPAN_EQ(expectedEdges, actualBoundaryEdges);

  // Element barycenters
  EXPECT_NEAR_EQ(Real3(0.25_r, 0.75_r, 0.25_r), mesh.GetElementBarycenter(0));
  EXPECT_NEAR_EQ(Real3(0.75_r, 0.75_r, 0.75_r), mesh.GetElementBarycenter(1));
  EXPECT_NEAR_EQ(Real3(0.75_r, 0.25_r, 0.25_r), mesh.GetElementBarycenter(2));
  EXPECT_NEAR_EQ(Real3(0.25_r, 0.25_r, 0.75_r), mesh.GetElementBarycenter(3));
  EXPECT_NEAR_EQ(Real3(0.5_r, 0.5_r, 0.5_r), mesh.GetElementBarycenter(4));

  // Aabb
  EXPECT_NEAR_EQ(Real3(0_r, 0_r, 0_r), mesh.GetAabb().GetMin());
  EXPECT_NEAR_EQ(Real3(1_r, 1_r, 1_r), mesh.GetAabb().GetMax());
  EXPECT_NEAR_EQ(Real3(0.5_r, 0.5_r, 0.5_r), mesh.GetObb().GetCenter());
  EXPECT_NEAR_EQ(Real3(0.5_r, 0.5_r, 0.5_r), mesh.GetObb().GetHalfExtents());
  EXPECT_NEAR_EQ(Eye<3>(), mesh.GetObb().GetRotation());

  // Volume (of unit cube)
  EXPECT_NEAR_EQ(1_r, mesh.GetTotalMeasure());

  // Element volume
  EXPECT_NEAR_EQ(0.16666666_r, mesh.GetElementMeasure(0));
  EXPECT_NEAR_EQ(0.16666666_r, mesh.GetElementMeasure(1));
  EXPECT_NEAR_EQ(0.16666666_r, mesh.GetElementMeasure(2));
  EXPECT_NEAR_EQ(0.16666666_r, mesh.GetElementMeasure(3));
  EXPECT_NEAR_EQ(0.33333333_r, mesh.GetElementMeasure(4));

  // Mesh Barycenter
  EXPECT_NEAR_EQ(Real3(0.5_r, 0.5_r, 0.5_r), mesh.GetBarycenter());

  // Boundary faces (any order of the correct indicies will be accepted here)
  std::vector<Int3> expectedBoundaryFaces = {
      Int3{2, 3, 6},
      Int3{0, 2, 3},
      Int3{0, 2, 6},
      Int3{3, 6, 7},
      Int3{3, 5, 7},
      Int3{5, 6, 7},
      Int3{0, 1, 3},
      Int3{1, 3, 5},
      Int3{0, 1, 5},
      Int3{0, 4, 5},
      Int3{0, 4, 6},
      Int3{4, 5, 6},
  };
  std::vector<Int3> actualBoundaryFaces(
      mesh.GetBoundaryFacesConnectivity().begin(), mesh.GetBoundaryFacesConnectivity().end());
  EXPECT_EQ(12, expectedBoundaryFaces.size());
  EXPECT_EQ(12, actualBoundaryFaces.size());
  for (size_t i = 0; i < 12; ++i) {
    std::sort(expectedBoundaryFaces[i].begin(), expectedBoundaryFaces[i].end());
    std::sort(actualBoundaryFaces[i].begin(), actualBoundaryFaces[i].end());
  }
  auto int3_less = [](Int3 const& a, Int3 const& b) {
    return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
  };
  std::sort(expectedBoundaryFaces.begin(), expectedBoundaryFaces.end(), int3_less);
  std::sort(actualBoundaryFaces.begin(), actualBoundaryFaces.end(), int3_less);
  EXPECT_SPAN_EQ(MakeSpan(expectedBoundaryFaces), MakeSpan(actualBoundaryFaces));

  // The test above did not consider the winding order. Now, lets verify that all the
  // surface faces have a normal that points away from the center.
  for (Int3 const& face : mesh.GetBoundaryFacesConnectivity()) {
    Real3 const faceNormal = Normalize(Cross(
        coordinates[face[1]] - coordinates[face[0]], coordinates[face[2]] - coordinates[face[1]]));
    Real3 const centerToFace = coordinates[face[0]] - mesh.GetBarycenter();
    EXPECT_LT(0_r, Dot(faceNormal, centerToFace));
  }

  // Boundary face adjacency (list of boundary faces that share a vertex)
  Span<Int3 const> boundaryFaceConnectivity = mesh.GetBoundaryFacesConnectivity();
  for (int iNode = 0; iNode < mesh.GetNumNodes(); ++iNode) {
    Span<int const> adjacentBoundaryFaces = mesh.GetAdjacentBoundaryFaces(iNode);
    for (int iFace = 0; iFace < (int)boundaryFaceConnectivity.size(); ++iFace) {
      Int3 const& nodesInFace = boundaryFaceConnectivity[iFace];
      bool expectFaceContainsNode =
          (std::find(adjacentBoundaryFaces.begin(), adjacentBoundaryFaces.end(), iFace) !=
           adjacentBoundaryFaces.end());
      bool faceActuallyContainsNode =
          (nodesInFace[0] == iNode) || (nodesInFace[1] == iNode) || (nodesInFace[2] == iNode);
      EXPECT_EQ(expectFaceContainsNode, faceActuallyContainsNode);
    }
  }

  // Vert adjacency (list of other verts that share an edge).
  // NOTE: Verts 1, 2, 4, & 7 are the tips of the corner tets. They are only used by one
  //       tet each. The other verts are shared between two more more tets.
  std::vector<int> expectedAdjacency[8] = {
      {1, 2, 3, 4, 5, 6},
      {0, 3, 5},
      {0, 3, 6},
      {0, 1, 2, 5, 6, 7},
      {0, 5, 6},
      {0, 1, 3, 4, 6, 7},
      {0, 2, 3, 4, 5, 7},
      {3, 5, 6},
  };
  for (int i = 0; i < mesh.GetNumElements(); ++i) {
    EXPECT_TRUE(test::EqualSpanUnordered(expectedAdjacency[i], mesh.GetAdjacentNodes(i)));
  }
}

TEST(TetrahedralMesh, UnitGridDefaultMatchesUnitCube) {
  Real3 constexpr kScale{1.5_r, 2.0_r, 3.5_r};
  TetrahedralMesh cube = test::CreateMinimalTetMeshUnitCube(kScale);
  TetrahedralMesh grid = test::CreateMinimalTetMeshUnitGrid(kScale);

  EXPECT_SPAN_EQ(cube.GetNodeCoordinates(), grid.GetNodeCoordinates());
  EXPECT_SPAN_EQ(cube.GetFlatConnectivity(), grid.GetFlatConnectivity());
}

TEST(TetrahedralMesh, ActiveNodeEdgesUseActiveNodeIndices) {
  constexpr std::array kCoordinates = {
      Real3{10_r, 10_r, 10_r}, // unreferenced
      Real3{0_r, 0_r, 0_r},
      Real3{1_r, 0_r, 0_r},
      Real3{20_r, 20_r, 20_r}, // unreferenced
      Real3{0_r, 1_r, 0_r},
      Real3{0_r, 0_r, 1_r},
  };
  constexpr std::array kConnectivity = {Int4{1, 2, 4, 5}};
  TetrahedralMesh const mesh(MakeConstSpan(kCoordinates), MakeConstSpan(kConnectivity));

  constexpr std::array kExpectedActiveNodes = {1, 2, 4, 5};
  constexpr std::array kExpectedEdges = {
      Int2{1, 2}, Int2{1, 4}, Int2{1, 5}, Int2{2, 4}, Int2{2, 5}, Int2{4, 5}};
  constexpr std::array kExpectedActiveEdges = {
      Int2{0, 1}, Int2{0, 2}, Int2{0, 3}, Int2{1, 2}, Int2{1, 3}, Int2{2, 3}};

  EXPECT_SPAN_EQ(kExpectedActiveNodes, mesh.GetActiveNodes());
  EXPECT_SPAN_EQ(kExpectedEdges, mesh.GetEdges());
  EXPECT_SPAN_EQ(kExpectedActiveEdges, mesh.GetActiveNodesEdges());
}

TEST(TetrahedralMesh, Empty) {
  // An empty mesh isn't very useful, but it shouldn't crash
  // nor divide by 0.0.
  TetrahedralMesh mesh(Span<Real3 const>{nullptr, 0}, Span<Int4 const>{nullptr, 0});
  EXPECT_EQ(0, mesh.GetNumNodes());
  EXPECT_EQ(0, mesh.GetNumElements());
  EXPECT_EQ(4, mesh.GetNumNodesPerElement());
  EXPECT_EQ(0, mesh.GetNumEdges());
  EXPECT_EQ(0, mesh.GetNodeCoordinates().size());
  EXPECT_EQ(0, mesh.GetFlatConnectivity().size());
  EXPECT_EQ(0, mesh.GetElementConnectivity().size());
  EXPECT_EQ(0, mesh.GetBoundaryNodes().size());
  EXPECT_EQ(0, mesh.GetEdges().size());
  EXPECT_EQ(0, mesh.GetBoundaryEdges().size());
  EXPECT_EQ(0, mesh.GetBoundaryFacesConnectivity().size());
  EXPECT_NEAR_TOL(Aabb{}, mesh.GetAabb(), 0_r); // zero tolerance
  EXPECT_NEAR_TOL(Obb{}, mesh.GetObb(), 0_r); // zero tolerance
  EXPECT_EQ(0_r, mesh.GetTotalMeasure());
  EXPECT_EQ(Real3{}, mesh.GetBarycenter());
}
