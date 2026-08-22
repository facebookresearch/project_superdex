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

#include <mochi_core/contact/contact_utils.h>
#include <mochi_core/geometry/grid_sdf.h>
#include <mochi_core/geometry/tetrahedral_map.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/nd_array_utils.h>
#include <mochi_core/utils/simd.h>
#include <mochi_core/utils/span.h>
#include <mochi_core/utils/vmatrix.h>

#include <gtest/gtest.h>

#include <numeric>
#include <utility>

using namespace mochi;

static std::pair<DynamicArray<Real3>, DynamicArray<int>> CreateTestData() {
  static constexpr int kNumPointsSide = 11;
  static constexpr real kNumPointsSideR = static_cast<real>(kNumPointsSide);
  static constexpr int kNumPoints = kNumPointsSide * kNumPointsSide * kNumPointsSide;
  DynamicArray<Real3> points(kNumPoints);

  // The sample points will be on a unit grid spanning from -1.1 to 1.627 in each dimension. These
  // will be used to test a mesh with nodes that span from 0 to 1 in each dimension. The sample
  // points are offset by 0.1 to avoid interior diagonals where minor differences in float precision
  // may result in major differences in the normal of the "closest face".
  Real3 minPoint = {-1.1_r, -1.1_r, -1.1_r};
  real delta = 3.0_r / kNumPointsSideR;
  for (int i = 0, p = 0; i < kNumPointsSide; i++) {
    for (int j = 0; j < kNumPointsSide; j++) {
      for (int k = 0; k < kNumPointsSide; k++, p++) {
        auto coord = StaticCast<Real3>(Int3{i, j, k});
        points[p] = minPoint + coord * delta;
      }
    }
  }

  DynamicArray<int> inds(kNumPoints);
  std::iota(inds.begin(), inds.end(), 0);

  return {points, inds};
}

TEST(TetrahedralMap, TetrahedralMap) {
  // Create meshes
  auto&& [coords, tetConnectivity] = test::CreateMinimalTetMeshUnitCube();
  auto&& [coordsDummy, triConnectivity] = test::CreateMinimalTriMeshUnitCube();
  auto tets = std::make_shared<TetrahedralMesh>(coords, tetConnectivity);
  auto tris = std::make_shared<TriangularMesh>(coords, triConnectivity);

  // Create a mesh collider from the surface mesh
  MeshCollider collider(tris);
  collider.Initialize();

  // Create the mapping
  TetrahedralMap tetMap(tets);

  // Define a grid of points
  auto [points, inds] = CreateTestData();

  // Query collision against the mesh collider directly
  ContactDetectionParams params;
  params.tolerance = 0_r;
  ContactDetectionResult collResult;
  FindPointContactsT(
      points,
      &collider,
      params,
      TransformRT{},
      collResult.sampleIndices,
      collResult.posColliding,
      collResult.sdfInfo,
      collResult.isSdfGradUnitary);

  // Map points with and without input BVH
  ContactDetectionResult mappedResult;
  DynamicArray<int> indsPoints;
  DynamicArray<Real3> pointsMapped;
  DynamicArray<Real3> pointsMappedTest;
  tetMap.MapPoints(
      points,
      inds,
      nullptr,
      pointsMapped,
      indsPoints,
      &mappedResult.jacColliderFromWorld,
      &mappedResult.jacWorldFromDofs);
  PointSetAabbObject bvhObject(points);
  BvhTreeParams bvhParams;
  bvhParams.splittingAlgorithm = BvhSplittingAlgorithm::TopDown_Mean;
  AabbTree bvh(&bvhObject, bvhParams);
  tetMap.MapPoints(
      points,
      inds,
      &bvh,
      pointsMappedTest,
      indsPoints,
      &mappedResult.jacColliderFromWorld,
      &mappedResult.jacWorldFromDofs);
  EXPECT_TRUE(test::EqualSpan(pointsMapped, pointsMappedTest));

  // Query collision against the mapped collider
  FindPointContactsT(
      pointsMapped,
      &collider,
      params,
      TransformRT{},
      mappedResult.sampleIndices,
      mappedResult.posColliding,
      mappedResult.sdfInfo,
      mappedResult.isSdfGradUnitary);
  BaseMap::ReindexResult(
      indsPoints,
      mappedResult.sampleIndices,
      &mappedResult.jacColliderFromWorld,
      &mappedResult.jacWorldFromDofs);
  mappedResult.isSdfGradUnitary = false;

  // Test if the colliding points are the same and have the same distance and normal.
  EXPECT_EQ(collResult.sampleIndices.size(), mappedResult.sampleIndices.size());
  if (collResult.sampleIndices.size() == mappedResult.sampleIndices.size()) {
    for (int i = 0; i < collResult.sampleIndices.size(); ++i) {
      EXPECT_EQ(collResult.sampleIndices[i], mappedResult.sampleIndices[i]);
      EXPECT_NEAR_EQ(collResult.sdfInfo.val[i], mappedResult.sdfInfo.val[i]);
      real normalDot = Dot(collResult.sdfInfo.grad[i], mappedResult.sdfInfo.grad[i]);
      EXPECT_NEAR(1_r, normalDot, 1e-9_r);
    }
  }

  // Move the meshes
  TransformRT transform(
      Quaternion::FromRotationVector(Vec4r(0.3_r, 0.2_r, 0.6_r)), Real3(-0.3_r, 0.5_r, 0.1_r));
  DynamicArray<Real3> disp(coords.size());
  for (int i = 0, j = 0; i < coords.size(); i++, j += 3) {
    disp[i] = coords[i];
    coords[i] = transform.TransformPoint(coords[i]);
    disp[i] = coords[i] - disp[i];
  }

  // Recompute the mesh collider
  auto tris2 = std::make_shared<TriangularMesh>(coords, triConnectivity);
  MeshCollider collider2(tris2);
  collider2.Initialize();

  // Update the map
  auto dofs = Flatten(MakeSpan(disp));
  tetMap.UpdateMap(dofs);

  // Query collision against the new mesh collider
  collResult = {};
  FindPointContactsT(
      points,
      &collider2,
      params,
      TransformRT{},
      collResult.sampleIndices,
      collResult.posColliding,
      collResult.sdfInfo,
      collResult.isSdfGradUnitary);

  // Query collision against the mapped mesh collider
  mappedResult = {};
  indsPoints = {};
  pointsMapped = {};
  tetMap.MapPoints(
      points,
      inds,
      nullptr,
      pointsMapped,
      indsPoints,
      &mappedResult.jacColliderFromWorld,
      &mappedResult.jacWorldFromDofs);
  FindPointContactsT(
      pointsMapped,
      &collider,
      params,
      TransformRT{},
      mappedResult.sampleIndices,
      mappedResult.posColliding,
      mappedResult.sdfInfo,
      mappedResult.isSdfGradUnitary);
  // Gradients are expressed in the local frame. Map them to match the result.
  for (int i = 0; i < mappedResult.sampleIndices.size(); i++) {
    int originalIndex = mappedResult.sampleIndices[i];
    mappedResult.sdfInfo.grad[i] = ToReal3(DotVecMat3x3(
        ToSimd(mappedResult.sdfInfo.grad[i]), mappedResult.jacColliderFromWorld[originalIndex]));
  }
  BaseMap::ReindexResult(
      indsPoints,
      mappedResult.sampleIndices,
      &mappedResult.jacColliderFromWorld,
      &mappedResult.jacWorldFromDofs);
  mappedResult.isSdfGradUnitary = false;

  // Test if the colliding points are the same and have the same distance and normal.
  EXPECT_EQ(collResult.sampleIndices.size(), mappedResult.sampleIndices.size());
  if (collResult.sampleIndices.size() == mappedResult.sampleIndices.size()) {
    for (int i = 0; i < collResult.sampleIndices.size(); ++i) {
      EXPECT_EQ(collResult.sampleIndices[i], mappedResult.sampleIndices[i]);
      EXPECT_NEAR_EQ(collResult.sdfInfo.val[i], mappedResult.sdfInfo.val[i]);
      real normalDot = Dot(collResult.sdfInfo.grad[i], mappedResult.sdfInfo.grad[i]);
      EXPECT_NEAR(1_r, normalDot, 1e-6_r);
    }
  }
}

TEST(TetrahedralMap, MapPointsDeduplicatesSharedFaceHits) {
  auto&& [coords, tetConnectivity] = test::CreateMinimalTetMeshTwoShareFace();
  auto tets = std::make_shared<TetrahedralMesh>(coords, tetConnectivity);
  TetrahedralMap tetMap(tets);

  // This point lies exactly on the shared face: x + y + z = 1.
  DynamicArray<Real3> const points = {Real3{0.5_r, 0.25_r, 0.25_r}};
  DynamicArray<int> const inds = {42};
  DynamicArray<Real3> mappedPoints;
  DynamicArray<int> mappedInds;
  DynamicArray<VMatrix3x3r> mapJac;
  DynamicArray<ColliderJacDofs> dofsJac;
  tetMap.MapPoints(points, inds, nullptr, mappedPoints, mappedInds, &mapJac, &dofsJac);

  EXPECT_EQ(1, mapJac.size());
  EXPECT_EQ(1, dofsJac.size());
  EXPECT_SPAN_EQ(inds, mappedInds);
  ASSERT_EQ(1, mappedPoints.size());
  EXPECT_NEAR_EQ(points[0], mappedPoints[0]);
}
