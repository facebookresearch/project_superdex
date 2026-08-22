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
#include <mochi_core/geometry/triangular_mesh.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/nd_array_utils.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

using namespace mochi;

TEST(TriangularMesh, Cube) {
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
  std::vector<Int3> const connectivity = {
      Int3{1, 3, 5}, // +x face
      Int3{3, 7, 5},
      Int3{5, 7, 4}, // +z face
      Int3{7, 6, 4},
      Int3{4, 6, 0}, // -x face
      Int3{6, 2, 0},
      Int3{0, 2, 1}, // -z face
      Int3{2, 3, 1},
      Int3{2, 6, 3}, // +y face
      Int3{6, 7, 3},
      Int3{4, 0, 5}, // -y face
      Int3{0, 1, 5},
  };

  TriangularMesh mesh(coordinates, connectivity);
  static_assert(3 == mesh.kSpaceDimension);
  static_assert(3 == mesh.kNodesPerElement);

  EXPECT_EQ(8, mesh.GetNumNodes());
  EXPECT_EQ(18, mesh.GetNumEdges());
  EXPECT_EQ(12, mesh.GetNumFaces()); // Same as GetNumElements for a TriangularMesh
  EXPECT_EQ(0, mesh.GetNumVolumes());
  EXPECT_EQ(12, mesh.GetNumElements());
  EXPECT_EQ(3, mesh.GetNumNodesPerElement());

  // Node coordinates
  auto nodeCoords = mesh.GetNodeCoordinates();
  EXPECT_SPAN_EQ(coordinates, nodeCoords);

  // Element connectivity (flat list)
  auto flatConnectivity = mesh.GetFlatConnectivity();
  EXPECT_SPAN_EQ(connectivity, Unflatten<Int3 const>(flatConnectivity));

  // Element connectivity (grouped by Int3)
  auto elementConnectivity = mesh.GetElementConnectivity();
  EXPECT_SPAN_EQ(connectivity, elementConnectivity);

  // Boundary nodes (none because it is a closed triangle mesh)
  EXPECT_EQ(0, mesh.GetBoundaryNodes().size());

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
      Int2{3, 5},
      Int2{4, 7},
      Int2{0, 6},
      Int2{1, 2},
      Int2{3, 6},
      Int2{0, 5},
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

  // Boundary edges (none because it is a closed triangle)
  EXPECT_EQ(0, mesh.GetBoundaryEdges().size());

  // Element barycenters
  std::vector<Real3> const expectedElementBarycenters = {
      Real3{1.00000000_r, 0.33333333_r, 0.33333333_r},
      Real3{1.00000000_r, 0.66666667_r, 0.66666666_r},
      Real3{0.66666667_r, 0.33333333_r, 1.00000000_r},
      Real3{0.33333333_r, 0.66666667_r, 1.00000000_r},
      Real3{0.00000000_r, 0.33333333_r, 0.66666667_r},
      Real3{0.00000000_r, 0.66666667_r, 0.33333333_r},
      Real3{0.33333333_r, 0.33333333_r, 0.00000000_r},
      Real3{0.66666667_r, 0.66666667_r, 0.00000000_r},
      Real3{0.33333333_r, 1.00000000_r, 0.33333333_r},
      Real3{0.66666667_r, 1.00000000_r, 0.66666667_r},
      Real3{0.33333333_r, 0.00000000_r, 0.66666667_r},
      Real3{0.66666667_r, 0.00000000_r, 0.33333333_r},
  };
  EXPECT_EQ(expectedElementBarycenters.size(), mesh.GetNumElements());
  for (int e = 0; e < mesh.GetNumElements(); ++e) {
    EXPECT_NEAR_EQ(expectedElementBarycenters[e], mesh.GetElementBarycenter(e));
  }

  // Aabb
  EXPECT_NEAR_EQ(Real3(0_r, 0_r, 0_r), mesh.GetAabb().GetMin());
  EXPECT_NEAR_EQ(Real3(1_r, 1_r, 1_r), mesh.GetAabb().GetMax());
  EXPECT_NEAR_EQ(Real3(0.5_r, 0.5_r, 0.5_r), mesh.GetObb().GetCenter());
  EXPECT_NEAR_EQ(Real3(0.5_r, 0.5_r, 0.5_r), mesh.GetObb().GetHalfExtents());
  EXPECT_NEAR_EQ(Eye<3>(), mesh.GetObb().GetRotation());

  // Element normals
  std::vector<Real3> const expected_element_normal = {
      Real3{+1_r, +0_r, +0_r}, // +x face
      Real3{+1_r, +0_r, +0_r},
      Real3{+0_r, +0_r, +1_r}, // +z face
      Real3{+0_r, +0_r, +1_r},
      Real3{-1_r, +0_r, +0_r}, // -x face
      Real3{-1_r, +0_r, +0_r},
      Real3{+0_r, +0_r, -1_r}, // -z face
      Real3{+0_r, +0_r, -1_r},
      Real3{+0_r, +1_r, +0_r}, // +y face
      Real3{+0_r, +1_r, +0_r},
      Real3{+0_r, -1_r, +0_r}, // -y face
      Real3{+0_r, -1_r, +0_r},
  };
  EXPECT_SPAN_EQ(expected_element_normal, mesh.GetElementNormals());

  // Area (of unit cube)
  EXPECT_NEAR_EQ(6_r, mesh.GetTotalMeasure());

  // Element area (half a unit square each)
  for (int i = 0; i < mesh.GetNumElements(); ++i) {
    EXPECT_NEAR_EQ(0.5_r, mesh.GetElementMeasure(i));
  }

  // Mesh Barycenter
  EXPECT_NEAR_EQ(Real3(0.5_r, 0.5_r, 0.5_r), mesh.GetBarycenter());
}

TEST(TriangularMesh, Empty) {
  // An empty triangle mesh isn't very useful, but it shouldn't crash
  // nor divide by 0.0.
  TriangularMesh mesh(Span<Real3 const>{nullptr, 0}, Span<Int3 const>{nullptr, 0});
  EXPECT_EQ(0, mesh.GetNumNodes());
  EXPECT_EQ(0, mesh.GetNumEdges());
  EXPECT_EQ(0, mesh.GetNumFaces());
  EXPECT_EQ(0, mesh.GetNumVolumes());
  EXPECT_EQ(0, mesh.GetNumElements());
  EXPECT_EQ(3, mesh.GetNumNodesPerElement());
  EXPECT_EQ(0, mesh.GetNodeCoordinates().size());
  EXPECT_EQ(0, mesh.GetFlatConnectivity().size());
  EXPECT_EQ(0, mesh.GetElementConnectivity().size());
  EXPECT_EQ(0, mesh.GetBoundaryNodes().size());
  EXPECT_EQ(0, mesh.GetEdges().size());
  EXPECT_EQ(0, mesh.GetBoundaryEdges().size());
  EXPECT_NEAR_TOL(Aabb{}, mesh.GetAabb(), 0_r); // zero tolerance
  EXPECT_NEAR_TOL(Obb{}, mesh.GetObb(), 0_r); // zero tolerance
  EXPECT_EQ(0, mesh.GetElementNormals().size());
  EXPECT_EQ(0_r, mesh.GetTotalMeasure());
  EXPECT_EQ(Real3{}, mesh.GetBarycenter());
}

TEST(TriangularMesh, Single) {
  // An open mesh with one triangle
  std::vector<Real3> const coordinates = {
      {0_r, 0_r, 0_r},
      {1_r, 0_r, 0_r},
      {0_r, 1_r, 0_r},
  };
  std::vector<Int3> const connectivity = {Int3{0, 1, 2}};
  TriangularMesh mesh(coordinates, connectivity);

  EXPECT_EQ(3, mesh.GetNumNodes());
  EXPECT_EQ(1, mesh.GetNumElements());
  EXPECT_EQ(3, mesh.GetNumNodesPerElement());
  EXPECT_EQ(3, mesh.GetNumEdges());
  EXPECT_SPAN_EQ(coordinates, mesh.GetNodeCoordinates());
  EXPECT_SPAN_EQ(connectivity, mesh.GetElementConnectivity());

  std::vector<int> expectedBoundaryNodes = {0, 1, 2};
  EXPECT_TRUE(test::EqualSpanUnordered(MakeSpan(expectedBoundaryNodes), mesh.GetBoundaryNodes()));

  // Edges (in any order)
  std::vector<Int2> expectedEdges = {Int2{0, 1}, Int2{0, 2}, Int2{1, 2}};
  std::vector<Int2> actualEdges(mesh.GetEdges().begin(), mesh.GetEdges().end());
  for (Int2& edge : actualEdges) {
    std::sort(edge.begin(), edge.end());
  }

  auto int2_less = [](Int2 const& a, Int2 const& b) {
    return (a[0] < b[0]) || (a[0] == b[0] && a[1] < b[1]);
  };
  std::sort(actualEdges.begin(), actualEdges.end(), int2_less);
  EXPECT_SPAN_EQ(expectedEdges, actualEdges);

  // Boundary edges (all 3 of them)
  std::vector<Int2> actualBoundaryEdges(
      mesh.GetBoundaryEdges().begin(), mesh.GetBoundaryEdges().end());
  for (Int2& edge : actualBoundaryEdges) {
    std::sort(edge.begin(), edge.end());
  }
  std::sort(actualBoundaryEdges.begin(), actualBoundaryEdges.end(), int2_less);
  EXPECT_SPAN_EQ(expectedEdges, actualBoundaryEdges);

  EXPECT_EQ(Real3(0_r, 0_r, 0_r), mesh.GetAabb().GetMin());
  EXPECT_EQ(Real3(1_r, 1_r, 0_r), mesh.GetAabb().GetMax());
  EXPECT_EQ(1, mesh.GetElementNormals().size());
  EXPECT_NEAR_EQ((Real3(0_r, 0_r, 1_r)), mesh.GetElementNormals()[0]);
  EXPECT_EQ(0.5_r, mesh.GetTotalMeasure());
  EXPECT_NEAR_EQ(Real3(0.33333333_r, 0.33333333_r, 0_r), mesh.GetBarycenter());
}

TEST(TriangularMesh, HalfEdgeStructure) {
  // An open mesh with a triforce
  std::vector<Real3> const coordinates = {
      Real3{0_r, 0_r, 0_r},
      Real3{1_r, 0_r, 0_r},
      Real3{2_r, 0_r, 0_r},
      Real3{0.5_r, 1_r, 0_r},
      Real3{1.5_r, 1_r, 0_r},
      Real3{1_r, 2_r, 0_r},
  };

  std::vector<Int3> const connectivity = {
      Int3{0, 1, 3}, Int3{1, 4, 3}, Int3{1, 2, 4}, Int3{3, 4, 5}};

  TriangularMesh mesh(coordinates, connectivity);
  auto halfEdge = mesh.GenerateHalfEdgeStructure();

  auto const& halfEdges = halfEdge.halfEdges;
  auto const& edge2half = halfEdge.edge2halfs;
  auto const& face2half = halfEdge.face2half;
  auto const& node2half = halfEdge.node2half;

  EXPECT_EQ(halfEdges.size(), 12);
  EXPECT_EQ(edge2half.size(), 9);
  EXPECT_EQ(face2half.size(), 4);
  EXPECT_EQ(node2half.size(), 6);

  // Check pair links
  int internalCount = 0;
  int boundaryCount = 0;
  for (size_t i = 0; i < halfEdges.size(); ++i) {
    if (halfEdges[i].pair != -1) {
      EXPECT_EQ(halfEdges[halfEdges[i].pair].pair, i);
      internalCount++;
    } else {
      boundaryCount++;
    }
  }

  EXPECT_EQ(internalCount, 6);
  EXPECT_EQ(boundaryCount, 6);

  // Check face links
  EXPECT_EQ(face2half[0], 0);
  EXPECT_EQ(face2half[1], 3);
  EXPECT_EQ(face2half[2], 6);
  EXPECT_EQ(face2half[3], 9);

  // Check edge links
  EXPECT_EQ(edge2half[0][0], 0);
  EXPECT_EQ(edge2half[0][1], -1);
  EXPECT_EQ(edge2half[1][0], 1);
  EXPECT_EQ(edge2half[1][1], 5);
  EXPECT_EQ(edge2half[2][0], 2);
  EXPECT_EQ(edge2half[2][1], -1);
  EXPECT_EQ(edge2half[3][0], 3);
  EXPECT_EQ(edge2half[3][1], 8);
  EXPECT_EQ(edge2half[4][0], 4);
  EXPECT_EQ(edge2half[4][1], 9);
  EXPECT_EQ(edge2half[5][0], 6);
  EXPECT_EQ(edge2half[5][1], -1);
  EXPECT_EQ(edge2half[6][0], 7);
  EXPECT_EQ(edge2half[6][1], -1);
  EXPECT_EQ(edge2half[7][0], 10);
  EXPECT_EQ(edge2half[7][1], -1);
  EXPECT_EQ(edge2half[8][0], 11);
  EXPECT_EQ(edge2half[8][1], -1);

  // Check node links
  EXPECT_EQ(node2half[0], 0);
  EXPECT_EQ(node2half[1], 1);
  EXPECT_EQ(node2half[2], 7);
  EXPECT_EQ(node2half[3], 2);
  EXPECT_EQ(node2half[4], 4);
  EXPECT_EQ(node2half[5], 11);

  // Check circle half-edge links
  for (int first : face2half) {
    EXPECT_EQ(halfEdges[halfEdges[halfEdges[first].next].next].next, first);
  }

  // Check circle node links
  for (size_t i = 0; i < face2half.size(); ++i) {
    int curr = face2half[i];
    EXPECT_EQ(halfEdges[curr].node, connectivity[i][1]);
    curr = halfEdges[curr].next;
    EXPECT_EQ(halfEdges[curr].node, connectivity[i][2]);
    curr = halfEdges[curr].next;
    EXPECT_EQ(halfEdges[curr].node, connectivity[i][0]);
  }
}
