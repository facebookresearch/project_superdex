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

#include "mesh_test_helpers.h"

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_mesh/mesh_statistics.h>

#include <cmath>

using namespace mochi;
using namespace mochi::mesh;
using namespace mochi::test;

TEST(MeshStatistics, CubeMesh) {
  MeshData const inputMesh = CreateCubeTriMesh();

  ExpectOK expectOK;
  MeshStatistics const stats = ComputeMeshStatistics(inputMesh, nullptr, expectOK);

  EXPECT_EQ(stats.numVertices, 8);
  EXPECT_EQ(stats.numFaces, 12);
  EXPECT_NEAR_TOL(stats.edgeLengths.min, 1.0, 1e-10);
  EXPECT_NEAR_TOL(stats.edgeLengths.max, std::sqrt(2.0), 1e-10);
  EXPECT_GT(stats.angles.min, 0.0);
  EXPECT_LT(stats.angles.max, 180.0);
  EXPECT_TRUE(stats.isClosed);
  EXPECT_EQ(stats.hausdorffDistance, -1.0);
}

TEST(MeshStatistics, CubeAngles) {
  MeshData const inputMesh = CreateCubeTriMesh();

  ExpectOK expectOK;
  MeshStatistics const stats = ComputeMeshStatistics(inputMesh, nullptr, expectOK);

  // The cube triangulation has right-angled isosceles triangles: 45-45-90 degrees.
  EXPECT_NEAR_TOL(stats.angles.min, 45.0, 1e-5);
  EXPECT_NEAR_TOL(stats.angles.max, 90.0, 1e-5);
}

TEST(MeshStatistics, WithReferenceMesh) {
  MeshData const inputMesh = CreateCubeTriMesh();
  MeshDataView const refView = inputMesh;

  ExpectOK expectOK;
  MeshStatistics const stats = ComputeMeshStatistics(inputMesh, &refView, expectOK);

  EXPECT_GE(stats.hausdorffDistance, 0.0);
  EXPECT_NEAR_TOL(stats.hausdorffDistance, 0.0, 1e-10);
}

TEST(MeshStatistics, OpenMesh_IsClosedFalse) {
  MeshData const inputMesh = CreateCubeTriMeshWithHole();

  ExpectOK expectOK;
  MeshStatistics const stats = ComputeMeshStatistics(inputMesh, nullptr, expectOK);

  EXPECT_FALSE(stats.isClosed);
}

TEST(MeshStatistics, InvalidReferenceMesh_NotTriangleMesh) {
  MeshData const inputMesh = CreateCubeTriMesh();
  MeshData refMesh;
  refMesh.nodesPerElement = 4;
  refMesh.coordinates = {0_r, 0_r, 0_r, 1_r, 0_r, 0_r, 0_r, 1_r, 0_r, 0_r, 0_r, 1_r};
  refMesh.connectivity = {0, 1, 2, 3};
  MeshDataView const refView = refMesh;

  ExpectNotOK expectNotOK;
  (void)ComputeMeshStatistics(inputMesh, &refView, expectNotOK);
}

TEST(MeshStatistics, InvalidReferenceMesh_EmptyCoordinates) {
  MeshData const inputMesh = CreateCubeTriMesh();
  MeshData refMesh;
  refMesh.nodesPerElement = 3;
  refMesh.connectivity = {0, 1, 2};
  MeshDataView const refView = refMesh;

  ExpectNotOK expectNotOK;
  (void)ComputeMeshStatistics(inputMesh, &refView, expectNotOK);
}

TEST(MeshStatistics, InvalidReferenceMesh_EmptyConnectivity) {
  MeshData const inputMesh = CreateCubeTriMesh();
  MeshData refMesh;
  refMesh.nodesPerElement = 3;
  refMesh.coordinates = {0_r, 0_r, 0_r, 1_r, 0_r, 0_r, 0_r, 1_r, 0_r};
  MeshDataView const refView = refMesh;

  ExpectNotOK expectNotOK;
  (void)ComputeMeshStatistics(inputMesh, &refView, expectNotOK);
}

TEST(MeshStatistics, InvalidInput_NotTriangleMesh) {
  MeshData inputMesh;
  inputMesh.nodesPerElement = 4;
  inputMesh.coordinates = {0_r, 0_r, 0_r, 1_r, 0_r, 0_r, 0_r, 1_r, 0_r, 0_r, 0_r, 1_r};
  inputMesh.connectivity = {0, 1, 2, 3};

  ExpectNotOK expectNotOK;
  (void)ComputeMeshStatistics(inputMesh, nullptr, expectNotOK);
}

TEST(MeshStatistics, InvalidInput_EmptyCoordinates) {
  MeshData inputMesh;
  inputMesh.nodesPerElement = 3;
  inputMesh.connectivity = {0, 1, 2};

  ExpectNotOK expectNotOK;
  (void)ComputeMeshStatistics(inputMesh, nullptr, expectNotOK);
}

TEST(MeshStatistics, InvalidInput_EmptyConnectivity) {
  MeshData inputMesh;
  inputMesh.nodesPerElement = 3;
  inputMesh.coordinates = {0_r, 0_r, 0_r, 1_r, 0_r, 0_r, 0_r, 1_r, 0_r};

  ExpectNotOK expectNotOK;
  (void)ComputeMeshStatistics(inputMesh, nullptr, expectNotOK);
}
