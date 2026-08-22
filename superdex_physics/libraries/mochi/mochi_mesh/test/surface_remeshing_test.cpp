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

#include <mochi_core/mochi_config.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_mesh/mesh_statistics.h>
#include <mochi_mesh/surface_remeshing.h>

using namespace mochi;
using namespace mochi::mesh;
using namespace mochi::test;

TEST(SurfaceRemeshing, CubeMesh) {
  MeshData const inputMesh = CreateCubeTriMesh();
  SurfaceRemeshingParams params;
  params.relativeToMeshSize = false;
  params.edgeSize = 1.0;

  ExpectOK expectOK;
  MeshData const result = RemeshSurface(inputMesh, params, expectOK);

  EXPECT_EQ(result.nodesPerElement, 3);
  EXPECT_GT(result.GetNumNodes(), 0);
  EXPECT_GT(result.GetNumElements(), 0);
  EXPECT_EQ(isize(result.coordinates) % 3, 0);
  EXPECT_EQ(isize(result.connectivity) % 3, 0);
}

TEST(SurfaceRemeshing, InvalidInput_NotTriangleMesh) {
  MeshData inputMesh;
  inputMesh.nodesPerElement = 4;
  inputMesh.coordinates = {0_r, 0_r, 0_r, 1_r, 0_r, 0_r, 0_r, 1_r, 0_r, 0_r, 0_r, 1_r};
  inputMesh.connectivity = {0, 1, 2, 3};

  SurfaceRemeshingParams params;
  ExpectNotOK expectNotOK;
  MeshData const result = RemeshSurface(inputMesh, params, expectNotOK);
}

TEST(SurfaceRemeshing, RemeshOpenMesh) {
  MeshData const inputMesh = CreateCubeTriMeshWithHole();
  SurfaceRemeshingParams params;
  params.relativeToMeshSize = false;
  params.edgeSize = 1.0;

  ExpectOK expectOK;
  MeshData const result = RemeshSurface(inputMesh, params, expectOK);

  EXPECT_EQ(result.nodesPerElement, 3);
  EXPECT_GT(result.GetNumNodes(), 0);
  EXPECT_GT(result.GetNumElements(), 0);
  EXPECT_EQ(isize(result.coordinates) % 3, 0);
  EXPECT_EQ(isize(result.connectivity) % 3, 0);
}

TEST(SurfaceRemeshing, ProtectConstraints) {
  MeshData const inputMesh = CreateCubeTriMesh();
  SurfaceRemeshingParams params;
  params.relativeToMeshSize = false;
  params.edgeSize = 1.0;
  params.protectConstraints = true;
  params.detectFeatures = true;

  ExpectOK expectOK;
  MeshData const result = RemeshSurface(inputMesh, params, expectOK);

  EXPECT_EQ(result.nodesPerElement, 3);
  EXPECT_GT(result.GetNumNodes(), 0);
  EXPECT_GT(result.GetNumElements(), 0);
}

TEST(SurfaceRemeshing, RelaxConstraints) {
  MeshData const inputMesh = CreateCubeTriMesh();
  SurfaceRemeshingParams params;
  params.relativeToMeshSize = false;
  params.edgeSize = 1.0;
  params.relaxConstraints = true;
  params.detectFeatures = true;

  ExpectOK expectOK;
  MeshData const result = RemeshSurface(inputMesh, params, expectOK);

  EXPECT_EQ(result.nodesPerElement, 3);
  EXPECT_GT(result.GetNumNodes(), 0);
  EXPECT_GT(result.GetNumElements(), 0);
}

TEST(SurfaceRemeshing, AdaptiveSizing) {
  MeshData const inputMesh = CreateCubeTriMesh();
  SurfaceRemeshingParams params;
  params.relativeToMeshSize = false;
  params.edgeSize = 1.0;
  params.useAdaptiveSizing = true;
  params.adaptiveSizingTolerance = 0.1;
  params.minEdgeSizeFactor = 0.25;
  params.maxEdgeSizeFactor = 2.0;

  ExpectOK expectOK;
  MeshData const result = RemeshSurface(inputMesh, params, expectOK);

  EXPECT_EQ(result.nodesPerElement, 3);
  EXPECT_GT(result.GetNumNodes(), 0);
  EXPECT_GT(result.GetNumElements(), 0);
}

TEST(SurfaceRemeshing, NoneMethodWithPostProcessing) {
  MeshData const inputMesh = CreateCubeTriMesh();
  SurfaceRemeshingParams params;
  params.method = RemeshMethod::None;
  params.relativeToMeshSize = false;
  params.edgeSize = 1.0;

  ExpectOK expectOK;
  MeshData const result = RemeshSurface(inputMesh, params, expectOK);

  EXPECT_EQ(result.nodesPerElement, 3);
  EXPECT_GT(result.GetNumNodes(), 0);
  EXPECT_GT(result.GetNumElements(), 0);
  EXPECT_EQ(isize(result.coordinates) % 3, 0);
  EXPECT_EQ(isize(result.connectivity) % 3, 0);
}

TEST(SurfaceRemeshing, RemeshWithRelativeToMeshSize) {
  MeshData const inputMesh = CreateCubeTriMesh();
  SurfaceRemeshingParams params;
  // relativeToMeshSize defaults to true — exercises bounding-box scaling path

  ExpectOK expectOK;
  MeshData const result = RemeshSurface(inputMesh, params, expectOK);

  EXPECT_EQ(result.nodesPerElement, 3);
  EXPECT_GT(result.GetNumNodes(), 0);
  EXPECT_GT(result.GetNumElements(), 0);
  EXPECT_EQ(isize(result.coordinates) % 3, 0);
  EXPECT_EQ(isize(result.connectivity) % 3, 0);
}

TEST(SurfaceRemeshing, InvalidInput_ZeroBoundingBoxExtent) {
  MeshData inputMesh;
  inputMesh.nodesPerElement = 3;
  inputMesh.coordinates = {0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r, 0_r};
  inputMesh.connectivity = {0, 1, 2};

  SurfaceRemeshingParams params;
  // relativeToMeshSize defaults to true — triggers zero bounding box extent error

  ExpectNotOK expectNotOK;
  MeshData const result = RemeshSurface(inputMesh, params, expectNotOK);
}

TEST(SurfaceRemeshing, InvalidInput_EmptyCoordinates) {
  MeshData inputMesh;
  inputMesh.nodesPerElement = 3;
  inputMesh.connectivity = {0, 1, 2};

  SurfaceRemeshingParams params;

  ExpectNotOK expectNotOK;
  MeshData const result = RemeshSurface(inputMesh, params, expectNotOK);
}

TEST(SurfaceRemeshing, InvalidInput_EmptyConnectivity) {
  MeshData inputMesh;
  inputMesh.nodesPerElement = 3;
  inputMesh.coordinates = {0_r, 0_r, 0_r, 1_r, 0_r, 0_r, 0_r, 1_r, 0_r};

  SurfaceRemeshingParams params;

  ExpectNotOK expectNotOK;
  MeshData const result = RemeshSurface(inputMesh, params, expectNotOK);
}

TEST(SurfaceRemeshing, InvalidInput_NegativeEdgeSize) {
  MeshData const inputMesh = CreateCubeTriMesh();
  SurfaceRemeshingParams params;
  params.edgeSize = -1.0;

  ExpectNotOK expectNotOK;
  (void)RemeshSurface(inputMesh, params, expectNotOK);
}

TEST(SurfaceRemeshing, InvalidInput_NegativeAlphaWrapAlpha) {
  MeshData const inputMesh = CreateCubeTriMesh();
  SurfaceRemeshingParams params;
  params.alphaWrapRelativeAlpha = -1.0;

  ExpectNotOK expectNotOK;
  (void)RemeshSurface(inputMesh, params, expectNotOK);
}

TEST(SurfaceRemeshing, AngleSmoothing) {
  MeshData const inputMesh = CreateCubeTriMesh();
  SurfaceRemeshingParams params;
  params.relativeToMeshSize = false;
  params.edgeSize = 1.0;
  params.tangentialRelaxationIterations = 0;
  params.angleSmoothingIterations = 3;

  ExpectOK expectOK;
  MeshData const result = RemeshSurface(inputMesh, params, expectOK);

  EXPECT_EQ(result.nodesPerElement, 3);
  EXPECT_GT(result.GetNumNodes(), 0);
  EXPECT_GT(result.GetNumElements(), 0);
}

TEST(SurfaceRemeshing, SurfaceDelaunayRemeshing_CubeMesh) {
  MeshData const cubeMesh = CreateCubeTriMesh();
  SurfaceRemeshingParams params;
  params.method = RemeshMethod::SurfaceDelaunay;
  params.edgeSize = 1.0;
  params.relativeToMeshSize = false;
  params.facetAngleBound = 20.0;
  params.sharpFeatureAngle = 60.0;

  ExpectOK expectOK;
  MeshData const result = RemeshSurface(cubeMesh, params, expectOK);

  EXPECT_EQ(result.nodesPerElement, 3);
  EXPECT_GT(result.GetNumNodes(), 0);
  EXPECT_GT(result.GetNumElements(), 0);

  MeshStatistics const stats = ComputeMeshStatistics(result, nullptr, expectOK);
  EXPECT_TRUE(stats.isClosed);
  EXPECT_GE(stats.angles.min, 15.0_r);
}

TEST(SurfaceRemeshing, AlphaWrapImproved_TangentialRelaxation) {
  MeshData const cubeMesh = CreateCubeTriMesh();
  SurfaceRemeshingParams params;
  params.method = RemeshMethod::AlphaWrap;
  params.edgeSize = 1.0;
  params.relativeToMeshSize = false;
  params.smoothingIterations = 3;
  params.tangentialRelaxationIterations = 5;
  params.angleSmoothingIterations = 0;
  params.relaxationStepsPerIteration = 3;

  ExpectOK expectOK;
  MeshData result = RemeshSurface(cubeMesh, params, expectOK);

  EXPECT_EQ(result.nodesPerElement, 3);
  EXPECT_GT(result.GetNumNodes(), 0);
  EXPECT_GT(result.GetNumElements(), 0);

  MeshStatistics const stats = ComputeMeshStatistics(result, nullptr, expectOK);
  EXPECT_TRUE(stats.isClosed);
}

TEST(SurfaceRemeshing, InvalidInput_NegativeSmoothingIterations) {
  MeshData const inputMesh = CreateCubeTriMesh();
  SurfaceRemeshingParams params;
  params.relativeToMeshSize = false;
  params.edgeSize = 1.0;
  params.smoothingIterations = -1;

  ExpectNotOK expectNotOK;
  (void)RemeshSurface(inputMesh, params, expectNotOK);
}

TEST(SurfaceRemeshing, InvalidInput_SharpFeatureAngleOutOfRange) {
  MeshData const inputMesh = CreateCubeTriMesh();
  SurfaceRemeshingParams params;
  params.relativeToMeshSize = false;
  params.edgeSize = 1.0;
  params.sharpFeatureAngle = 200.0;

  ExpectNotOK expectNotOK;
  (void)RemeshSurface(inputMesh, params, expectNotOK);
}

TEST(SurfaceRemeshing, InvalidInput_AdaptiveSizingMaxLessThanMin) {
  MeshData const inputMesh = CreateCubeTriMesh();
  SurfaceRemeshingParams params;
  params.relativeToMeshSize = false;
  params.edgeSize = 1.0;
  params.useAdaptiveSizing = true;
  params.minEdgeSizeFactor = 2.0;
  params.maxEdgeSizeFactor = 1.0;

  ExpectNotOK expectNotOK;
  (void)RemeshSurface(inputMesh, params, expectNotOK);
}
#if MOCHI_USE_EIGEN
TEST(SurfaceRemeshing, ACVDRemeshing_CubeMesh) {
  MeshData const cubeMesh = CreateCubeTriMesh();
  SurfaceRemeshingParams params;
  params.method = RemeshMethod::ACVD;
  params.edgeSize = 1.0;
  params.relativeToMeshSize = false;
  params.targetVertexCount = 50;
  params.acvdGradationFactor = 0.0;

  ExpectOK expectOK;
  MeshData result = RemeshSurface(cubeMesh, params, expectOK);

  EXPECT_EQ(result.nodesPerElement, 3);
  EXPECT_GT(result.GetNumNodes(), 0);
  EXPECT_GT(result.GetNumElements(), 0);

  MeshStatistics const stats = ComputeMeshStatistics(result, nullptr, expectOK);
  EXPECT_TRUE(stats.isClosed);
}

TEST(SurfaceRemeshing, ACVDRemeshing_NearTargetVertexCount) {
  MeshData const cubeMesh = CreateCubeTriMesh();
  SurfaceRemeshingParams params;
  params.method = RemeshMethod::ACVD;
  params.edgeSize = 1.0;
  params.relativeToMeshSize = false;
  params.targetVertexCount = 50;
  params.acvdGradationFactor = 0.0;
  // Disable post-processing so we directly measure ACVD vertex-count control.
  params.smoothingIterations = 0;
  params.tangentialRelaxationIterations = 0;
  params.angleSmoothingIterations = 0;
  params.repairMesh = false;

  ExpectOK expectOK;
  MeshData result = RemeshSurface(cubeMesh, params, expectOK);

  MeshStatistics const stats = ComputeMeshStatistics(result, nullptr, expectOK);
  // ACVD should hit the target vertex count within a modest tolerance.
  EXPECT_NEAR(stats.numVertices, 50, 20);
}

TEST(SurfaceRemeshing, ACVDRemeshing_RejectsNegativeTargetCount) {
  MeshData const cubeMesh = CreateCubeTriMesh();
  SurfaceRemeshingParams params;
  params.method = RemeshMethod::ACVD;
  params.relativeToMeshSize = false;
  params.edgeSize = 1.0;
  params.targetVertexCount = -1;

  ExpectNotOK expectNotOK;
  (void)RemeshSurface(cubeMesh, params, expectNotOK);
}
#else
TEST(SurfaceRemeshing, ACVDRemeshing_RequiresEigen) {
  MeshData const cubeMesh = CreateCubeTriMesh();
  SurfaceRemeshingParams params;
  params.method = RemeshMethod::ACVD;
  params.relativeToMeshSize = false;
  params.edgeSize = 1.0;

  ExpectNotOK expectNotOK;
  (void)RemeshSurface(cubeMesh, params, expectNotOK);
}
#endif

TEST(SurfaceRemeshing, SurfaceDelaunay_RejectsAngleBoundAbove30) {
  MeshData const cubeMesh = CreateCubeTriMesh();
  SurfaceRemeshingParams params;
  params.method = RemeshMethod::SurfaceDelaunay;
  params.relativeToMeshSize = false;
  params.edgeSize = 1.0;
  params.facetAngleBound = 31.0;

  ExpectNotOK expectNotOK;
  (void)RemeshSurface(cubeMesh, params, expectNotOK);
}

TEST(SurfaceRemeshing, SurfaceDelaunay_RejectsNegativeDistanceBound) {
  MeshData const cubeMesh = CreateCubeTriMesh();
  SurfaceRemeshingParams params;
  params.method = RemeshMethod::SurfaceDelaunay;
  params.relativeToMeshSize = false;
  params.edgeSize = 1.0;
  params.facetDistanceBound = -0.1;

  ExpectNotOK expectNotOK;
  (void)RemeshSurface(cubeMesh, params, expectNotOK);
}

TEST(SurfaceRemeshing, NoneMethod_InconsistentWinding) {
  MeshData const inputMesh = CreateCubeTriMeshWithInconsistentWinding();
  SurfaceRemeshingParams params;
  params.method = RemeshMethod::None;
  params.relativeToMeshSize = false;
  params.edgeSize = 1.0;

  ExpectOK expectOK;
  MeshData const result = RemeshSurface(inputMesh, params, expectOK);

  EXPECT_EQ(result.nodesPerElement, 3);
  EXPECT_GT(result.GetNumNodes(), 0);
  EXPECT_GT(result.GetNumElements(), 0);
}

TEST(SurfaceRemeshing, SurfaceDelaunay_InconsistentWinding) {
  MeshData const inputMesh = CreateCubeTriMeshWithInconsistentWinding();
  SurfaceRemeshingParams params;
  params.method = RemeshMethod::SurfaceDelaunay;
  params.relativeToMeshSize = false;
  params.edgeSize = 1.0;
  params.facetAngleBound = 20.0;
  params.sharpFeatureAngle = 60.0;

  ExpectOK expectOK;
  MeshData const result = RemeshSurface(inputMesh, params, expectOK);

  EXPECT_EQ(result.nodesPerElement, 3);
  EXPECT_GT(result.GetNumNodes(), 0);
  EXPECT_GT(result.GetNumElements(), 0);
}

TEST(SurfaceRemeshing, Statistics_InconsistentWinding) {
  MeshData const inputMesh = CreateCubeTriMeshWithInconsistentWinding();

  ExpectOK expectOK;
  MeshStatistics const stats = ComputeMeshStatistics(inputMesh, nullptr, expectOK);

  EXPECT_GT(stats.numVertices, 0);
  EXPECT_GT(stats.numFaces, 0);
  EXPECT_TRUE(stats.isClosed);
}

TEST(SurfaceRemeshing, SurfaceDelaunay_RelativeToMeshSize_Default) {
  // Exercises the default relativeToMeshSize=true path which has the unit-scaling logic.
  MeshData const cubeMesh = CreateCubeTriMesh();
  SurfaceRemeshingParams params;
  params.method = RemeshMethod::SurfaceDelaunay;
  // relativeToMeshSize defaults to true
  params.facetAngleBound = 20.0;

  ExpectOK expectOK;
  MeshData const result = RemeshSurface(cubeMesh, params, expectOK);

  EXPECT_EQ(result.nodesPerElement, 3);
  EXPECT_GT(result.GetNumNodes(), 0);
  EXPECT_GT(result.GetNumElements(), 0);
}
