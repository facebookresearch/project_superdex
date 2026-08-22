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

#include <gtest/gtest.h>
#include <mochi_core/contact/contact_utils.h>
#include <mochi_core/geometry/bvh_tree.h>
#include <mochi_core/geometry/geometry_utils.h>
#include <mochi_core/geometry/scalar_field.h>
#include <mochi_core/geometry/sdf_bv.h>
#include <mochi_core/geometry/triangular_mesh.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/nd_array_utils.h>
#include <mochi_core/utils/simd.h>
#include <mochi_core/utils/vmatrix.h>
#include <picojson/picojson.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "data/bvh_tree_sphere_cloud_data.h"

using namespace mochi;

/*************************************************************************************************/

template <typename Bv>
struct SphereCloudBvhObject : public BvhObject<Bv> {
  SphereCloudBvhObject(std::vector<Sphere> const& spheres) : spheres(spheres) {
    Update();
  }

  void Update() {
    bvs.resize(spheres.size());

    for (size_t index = 0; index < spheres.size(); ++index) {
      GetBoundingVolume(spheres[index], bvs[index]);
      bvs[index] = ExpandShape(bvs[index], 1e-7_r);
    }
  }

  int GetNumElements() const override {
    return isize(bvs);
  }

  Bv GetBv(int index) const override {
    return bvs[index];
  }

  real GetDistanceSqr(Real3 const& point, int index) const override {
    return Max(Get0(VDistancePointShapeSqr(ToSimd(point), spheres[index])), 0.0_r);
  }

  Vec4r VGetDistanceSqr(Vec4r point, int index) const override {
    return Max(VDistancePointShapeSqr(point, spheres[index]), SimdZero());
  }

  std::vector<Sphere> spheres;
  std::vector<Bv> bvs;
};

static NdArray<bool, 16, 16> DecodeContactMatrix(NdArray<unsigned int, 8> values) {
  NdArray<bool, 16, 16> matrix{};

  for (size_t i = 0; i < 8; ++i) {
    size_t c0 = values[i] & 0xFFFF;
    size_t c1 = (values[i] >> 16) & 0xFFFF;

    for (size_t j = 0; j < 16; ++j) {
      matrix[i * 2][j] = (c0 & 1);
      matrix[i * 2 + 1][j] = (c1 & 1);
      c0 >>= 1;
      c1 >>= 1;
    }
  }

  return matrix;
}

static NdArray<bool, 16, 16> BuildContactMatrix(std::vector<std::pair<int, int>> const& query) {
  NdArray<bool, 16, 16> matrix;

  // Initialize with identity.
  for (size_t i = 0; i < 16; ++i) {
    for (size_t j = 0; j < 16; ++j) {
      matrix[i][j] = i == j;
    }
  }

  // Fill in with intersection pairs.
  for (auto q : query) {
    MOCHI_ASSERT(q.first < 16);
    MOCHI_ASSERT(q.second < 16);
    matrix[q.first][q.second] = true;
    matrix[q.second][q.first] = true;
  }

  // Done!
  return matrix;
}

/*************************************************************************************************/

template <typename Bv>
void SphereCloud_Intersect() {
  size_t constexpr kNumCases = kSphereCloud_Points.dims[0];
  size_t constexpr kNumSpheres = kSphereCloud_Points.dims[1];

  // Build sphere cloud from data.
  std::vector<Sphere> spheresA(kNumSpheres);
  std::vector<Sphere> spheresB(kNumSpheres);
  for (size_t i = 0; i < kNumSpheres; ++i) {
    spheresA[i] = spheresB[i] = Sphere(kSphereCloud_Points[0][i], kSphereCloud_Radius);
  }

  // Build Bvh trees.
  SphereCloudBvhObject<Bv> objectA(spheresA);
  SphereCloudBvhObject<Bv> objectB(spheresB);

  BvhTreeParams treeParams{};
  treeParams.maxElementsPerLeaf = 1;
  treeParams.maxDepthPerBranch = 1000;

  BvhTree<Bv> treeA(&objectA, treeParams);
  BvhTree<Bv> treeB(&objectB, treeParams);

  // Perform overlap tests.
  for (size_t c0 = 0; c0 < kNumCases; ++c0) {
    for (size_t c1 = 0; c1 < kNumCases; ++c1) {
      // Refit spheres.
      for (size_t i = 0; i < kNumSpheres; ++i) {
        objectA.spheres[i] = Sphere(kSphereCloud_Points[c0][i], kSphereCloud_Radius);
        objectB.spheres[i] = Sphere(kSphereCloud_Points[c1][i], kSphereCloud_Radius);
      }
      objectA.Update();
      objectB.Update();
      treeA.Refit();
      treeB.Refit();

      // Query potential self-overlapping entries.
      std::vector<std::pair<int, int>> query;
      treeA.Intersect(treeB, [&query](int i, int j) { query.emplace_back(i, j); });

      // Build contact matrix.
      auto contact = BuildContactMatrix(query);
      auto expected = DecodeContactMatrix(kSphereCloud_OverlapTest[c0][c1]);

      // Check if intersecting spheres in ground truth are also present in the contact matrix. NOTE:
      // We only care about false negatives as the results of the intersection query are
      // approximate.
      for (size_t i = 0; i < kNumSpheres; ++i) {
        for (size_t j = i + 1; j < kNumSpheres; ++j) {
          if (expected[i][j]) {
            EXPECT_TRUE(contact[i][j]);
          }
        }
      }
    }
  }
}

TEST(BvhTree, SphereCloud_Intersect) {
  SphereCloud_Intersect<Aabb>();
  SphereCloud_Intersect<Sphere>();
}

template <typename Bv>
void SphereCloud_FindClosest() {
  size_t constexpr kNumCases = kSphereCloud_Points.dims[0];
  size_t constexpr kNumSpheres = kSphereCloud_Points.dims[1];

  // Build sphere cloud from data.
  std::vector<Sphere> spheres(kNumSpheres);
  for (size_t i = 0; i < kNumSpheres; ++i) {
    spheres[i] = Sphere(kSphereCloud_Points[0][i], kSphereCloud_Radius);
  }

  // Build Bvh tree.
  BvhTreeParams treeParams{};
  treeParams.maxElementsPerLeaf = 1;
  treeParams.maxDepthPerBranch = 1000;
  SphereCloudBvhObject<Bv> object(spheres);
  BvhTree<Bv> tree(&object, treeParams);

  // Perform all distance tests.
  for (size_t c0 = 0; c0 < kNumCases; ++c0) {
    // Refit spheres.
    for (size_t i = 0; i < kNumSpheres; ++i) {
      object.spheres[i] = Sphere(kSphereCloud_Points[c0][i], kSphereCloud_Radius);
    }
    object.Update();
    tree.Refit();

    // For all point clouds in the other test cases, find whichever is the closest.
    for (size_t c1 = 0; c1 < kNumCases; ++c1) {
      for (size_t i = 0; i < kNumSpheres; ++i) {
        auto const point = kSphereCloud_Points[c1][i];

        // Brute-force test.
        real bruteForceDistanceSqr = std::numeric_limits<real>::infinity();

        for (int j = 0; j < kNumSpheres; ++j) {
          auto distance =
              Max(Get0(VDistancePointShapeSqr(ToSimd(point), object.spheres[j])), 0.0_r);
          if (distance < bruteForceDistanceSqr) {
            bruteForceDistanceSqr = distance;
          }
        }

        // Aabb-test.
        real bvDistanceSqr = std::numeric_limits<real>::infinity();
        tree.FindClosest(point, &bvDistanceSqr);

        // Both distances should match.
        EXPECT_NEAR_RTOL(bruteForceDistanceSqr, bvDistanceSqr, 1e-3_r);
      }
    }
  }
}

TEST(BvhTree, SphereCloud_FindClosest) {
  SphereCloud_FindClosest<Aabb>();
  SphereCloud_FindClosest<Sphere>();
}

/*************************************************************************************************/
/* Tests for AABB tree of a tet mesh */

static std::vector<Real3> CreateTestData(int numPointsSide) {
  int const numPoints = numPointsSide * numPointsSide * numPointsSide;
  std::vector<Real3> points(numPoints);

  Real3 minPoint = {-1.0_r, -1.0_r, -1.0_r};
  real delta = 3.0_r / numPointsSide;
  for (int i = 0, p = 0; i < numPointsSide; i++) {
    for (int j = 0; j < numPointsSide; j++) {
      for (int k = 0; k < numPointsSide; k++, p++) {
        auto coord = StaticCast<Real3>(Int3{i, j, k});
        points[p] = minPoint + coord * delta;
      }
    }
  }

  return points;
}

TEST(BvhTree, TetMesh_AABBTree) {
  static constexpr int kNumPointsSide = 10;

  // Create meshes
  auto&& [coords, tetConnectivity] = test::CreateMinimalTetMeshUnitCube();
  auto&& [coordsDummy, triConnectivity] = test::CreateMinimalTriMeshUnitCube();

  auto tets = std::make_shared<TetrahedralMesh const>(coords, tetConnectivity);
  auto tris = std::make_shared<TriangularMesh const>(coords, triConnectivity);

  // Create a mesh collider from the surface mesh
  MeshCollider collider(tris);
  collider.Initialize();

  // Create an AABB tree. Make sure to use the dynamic vector of coordinates, to allow dynamic
  // refitting.
  TetrahedralMeshAabbObject aabbTreeObject(tets, coords);
  AabbTree aabbTree(&aabbTreeObject, BvhTreeParams{});

  // Define a grid of points
  auto points = CreateTestData(kNumPointsSide);

  // Query collision against the mesh collider
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

  // Query collision against the AABB tree
  ContactDetectionResult aabbResult;
  for (int i = 0; i < points.size(); ++i) {
    Vec4r position = ToSimd(points[i]);
    real closestDist = std::numeric_limits<real>::infinity();
    aabbTree.VFindClosest(position, &closestDist);
    if (closestDist == 0.0_r) {
      aabbResult.sampleIndices.push_back(i);
    }
  }

  // Test if the colliding points are the same
  EXPECT_EQ(collResult.sampleIndices.size(), aabbResult.sampleIndices.size());
  if (collResult.sampleIndices.size() == aabbResult.sampleIndices.size()) {
    for (int i = 0; i < collResult.sampleIndices.size(); ++i) {
      EXPECT_EQ(collResult.sampleIndices[i], aabbResult.sampleIndices[i]);
    }
  }

  // Move the meshes
  TransformRT transform(
      Quaternion::FromRotationVector(Vec4r(0.3_r, 0.2_r, 0.6_r)), Real3(-0.3_r, 0.5_r, 0.1_r));
  for (auto& i : coords) {
    i = transform.TransformPoint(i);
  }

  // Recompute the mesh collider
  auto tris2 = std::make_shared<TriangularMesh const>(coords, triConnectivity);
  MeshCollider collider2(tris2);
  collider2.Initialize();

  // Refit the AABB tree
  aabbTree.Refit();

  // Query collision against the mesh collider
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

  // Query collision against the AABB tree
  aabbResult = {};
  for (int i = 0; i < points.size(); ++i) {
    Vec4r position = ToSimd(points[i]);
    real closestDist = std::numeric_limits<real>::infinity();
    aabbTree.VFindClosest(position, &closestDist);
    if (closestDist <= 0.0_r) {
      aabbResult.sampleIndices.push_back(i);
    }
  }

  // Test if the colliding points are the same
  EXPECT_EQ(collResult.sampleIndices.size(), aabbResult.sampleIndices.size());
  if (collResult.sampleIndices.size() == aabbResult.sampleIndices.size()) {
    for (int i = 0; i < collResult.sampleIndices.size(); ++i) {
      EXPECT_EQ(collResult.sampleIndices[i], aabbResult.sampleIndices[i]);
    }
  }
}

/*************************************************************************************************/
/* Tests for AABB tree of a point set */

TEST(BvhTree, PointSet_AABBTree) {
  static constexpr int kNumPointsSide = 25;

  // Define a set of pseudo-random points
  std::vector<Real3> pointSet = {
      {5_r, -1_r, 2_r}, {-4_r, 1_r, -1_r}, {1_r, -3_r, 2_r}, {-3_r, -2_r, -4_r}, {5_r, -2_r, 4_r}};

  // Create an AABB tree.
  PointSetAabbObject aabbTreeObject(pointSet);
  BvhTreeParams bvhParams;
  bvhParams.splittingAlgorithm = BvhSplittingAlgorithm::TopDown_Mean;
  AabbTree aabbTree(&aabbTreeObject, bvhParams);

  // Define a grid of test points
  auto points = CreateTestData(kNumPointsSide);

  // Compute distances using the AABB tree
  std::vector<real> distSqr(points.size(), std::numeric_limits<real>::infinity());
  for (int i = 0; i < points.size(); i++) {
    aabbTree.VFindClosest(ToSimd(points[i]), &distSqr[i]);
  }

  // Compute distances using a brute-force test
  std::vector<real> distSqrTest(points.size(), std::numeric_limits<real>::infinity());
  for (int i = 0; i < points.size(); i++) {
    for (auto const& j : pointSet) {
      real distSqrThis = Get0(VNormSqr<3>(ToSimd(points[i]) - ToSimd(j)));
      if (distSqrThis < distSqrTest[i]) {
        distSqrTest[i] = distSqrThis;
      }
    }
  }

  // Test if the distances are the same. There may be floating point error due to conversions
  // between real and SIMD formats
  for (int i = 0; i < points.size(); ++i) {
    EXPECT_NEAR_RTOL(distSqr[i], distSqrTest[i], 1e-6_r);
  }

  // Move the points and refit the AABB tree
  for (auto& i : pointSet) {
    i = {i[1], i[2], i[0]};
  }
  aabbTree.Refit();

  // Compute distances using the AABB tree
  std::fill(distSqr.begin(), distSqr.end(), std::numeric_limits<real>::infinity());
  for (int i = 0; i < points.size(); i++) {
    aabbTree.VFindClosest(ToSimd(points[i]), &distSqr[i]);
  }

  // Compute distances using a brute-force test
  std::fill(distSqrTest.begin(), distSqrTest.end(), std::numeric_limits<real>::infinity());
  for (int i = 0; i < points.size(); i++) {
    for (auto const& j : pointSet) {
      real distSqrThis = Get0(VNormSqr<3>(ToSimd(points[i]) - ToSimd(j)));
      if (distSqrThis < distSqrTest[i]) {
        distSqrTest[i] = distSqrThis;
      }
    }
  }

  // Test if the distances are the same. There may be floating point error due to conversions
  // between real and SIMD formats
  for (int i = 0; i < points.size(); ++i) {
    EXPECT_NEAR_RTOL(distSqr[i], distSqrTest[i], 1e-6_r);
  }
}

TEST(BvhTree, PointSet_vs_TetMesh_AABBTree) {
  static constexpr int kNumPointsSide = 10;

  // Create a tet mesh
  auto&& [coordsAux, tetConnectivityAux] = test::CreateMinimalTetMeshUnitCube();
  auto& tetConnectivity = tetConnectivityAux;
  auto& coords = coordsAux;
  auto tets = std::make_shared<TetrahedralMesh const>(coords, tetConnectivity);

  // Create an AABB tree. Make sure to use the dynamic vector of coordinates, to allow dynamic
  // refitting.
  TetrahedralMeshAabbObject aabbTreeObjectTets(tets, coords);
  AabbTree aabbTreeTets(&aabbTreeObjectTets, BvhTreeParams{});

  // Create the point set
  auto pointSet = CreateTestData(kNumPointsSide);

  // Create an AABB tree.
  PointSetAabbObject aabbTreeObjectPoints(pointSet);
  BvhTreeParams bvhParams;
  bvhParams.splittingAlgorithm = BvhSplittingAlgorithm::TopDown_Mean;
  AabbTree aabbTreePoints(&aabbTreeObjectPoints, bvhParams);

  // Intersect the trees. Store the result as pairs of point and tet indices
  std::vector<std::pair<int, int>> result;
  result.reserve(pointSet.size());
  auto IntersectionCallback = [&result, &tetConnectivity, &coords, &pointSet](
                                  int tetIndex, int pointIndex) {
    Int4 tetInds = tetConnectivity[tetIndex];
    if (IsInsideTetrahedron(
            coords[tetInds[0]],
            coords[tetInds[1]],
            coords[tetInds[2]],
            coords[tetInds[3]],
            pointSet[pointIndex])) {
      result.emplace_back(pointIndex, tetIndex);
    }
  };
  aabbTreeTets.Intersect(aabbTreePoints, IntersectionCallback);

  // Sort the intersecting pairs for convenience of the test
  std::sort(result.begin(), result.end(), [](auto a, auto b) { return a.first < b.first; });

  // Brute-force pairwise testing
  std::vector<std::pair<int, int>> resultTest;
  resultTest.reserve(pointSet.size());
  for (int i = 0; i < pointSet.size(); i++) {
    [[maybe_unused]] bool isInside = false;
    for (int j = 0; j < tetConnectivity.size(); j++) {
      Int4 tetInds = tetConnectivity[j];
      if (IsInsideTetrahedron(
              coords[tetInds[0]],
              coords[tetInds[1]],
              coords[tetInds[2]],
              coords[tetInds[3]],
              pointSet[i])) {
        MOCHI_ASSERT_VERBOSE(!isInside, "The point is inside two tetrahedra");
        isInside = true;
        resultTest.emplace_back(i, j);
      }
    }
  }

  // Test if the results are the same
  EXPECT_EQ(result.size(), resultTest.size());
  if (result.size() == resultTest.size()) {
    for (int i = 0; i < result.size(); ++i) {
      EXPECT_EQ(result[i].first, resultTest[i].first);
      EXPECT_EQ(result[i].second, resultTest[i].second);
    }
  }
}

/*************************************************************************************************/
/* Tests for FindIntersectingElements method */

// Helper function to test FindIntersectingElements with a given query shape
template <typename Bv, typename QueryShape>
static void TestFindIntersectingElements(
    BvhTree<Bv>& tree,
    SphereCloudBvhObject<Bv>& object,
    QueryShape const& queryShape,
    size_t kNumSpheres) {
  // Test with kSkipElementBvCheck = false (precise)
  DynamicArray<int> preciseResults;
  tree.template FindIntersectingElements<false>(queryShape, preciseResults);

  // Test with kSkipElementBvCheck = true (faster but less precise)
  DynamicArray<int> fastResults;
  tree.template FindIntersectingElements<true>(queryShape, fastResults);

  // Brute force check to verify results
  std::vector<int> bruteForceResults;
  for (int i = 0; i < kNumSpheres; ++i) {
    if constexpr (std::is_same_v<QueryShape, AnyShape>) {
      if (std::visit(
              [&](auto const& shape) { return HasOverlap(shape, object.GetBv(i)); }, queryShape)) {
        bruteForceResults.push_back(i);
      }
    } else {
      if (HasOverlap(queryShape, object.GetBv(i))) {
        bruteForceResults.push_back(i);
      }
    }
  }
  ASSERT_FALSE(bruteForceResults.empty()); // Ensure not a dummy test.

  // Verify that precise results match brute force
  EXPECT_EQ(preciseResults.size(), bruteForceResults.size());
  for (int i = 0; i < isize(bruteForceResults); ++i) {
    bool found = false;
    for (int j = 0; j < isize(preciseResults); ++j) {
      if (bruteForceResults[i] == preciseResults[j]) {
        found = true;
        break;
      }
    }
    EXPECT_TRUE(found);
  }

  // Fast results may include false positives but should not have false negatives
  for (int i = 0; i < isize(bruteForceResults); ++i) {
    bool found = false;
    for (int j = 0; j < isize(fastResults); ++j) {
      if (bruteForceResults[i] == fastResults[j]) {
        found = true;
        break;
      }
    }
    EXPECT_TRUE(found);
  }
}

template <typename Bv>
static void SphereCloud_FindIntersectingElements() {
  auto constexpr kNumCases = kSphereCloud_Points.dims[0];
  auto constexpr kNumSpheres = kSphereCloud_Points.dims[1];

  // Build sphere cloud from data.
  std::vector<Sphere> spheres(kNumSpheres);
  for (int i = 0; i < kNumSpheres; ++i) {
    spheres[i] = Sphere(kSphereCloud_Points[0][i], kSphereCloud_Radius);
  }

  // Build Bvh tree.
  BvhTreeParams treeParams{};
  treeParams.maxElementsPerLeaf = 1;
  treeParams.maxDepthPerBranch = 1000;
  SphereCloudBvhObject<Bv> object(spheres);
  BvhTree<Bv> tree(&object, treeParams);

  // Perform tests for each case
  for (size_t c = 0; c < kNumCases; ++c) {
    // Refit spheres.
    for (size_t i = 0; i < kNumSpheres; ++i) {
      object.spheres[i] = Sphere(kSphereCloud_Points[c][i], kSphereCloud_Radius);
    }
    object.Update();
    tree.Refit();

    // Test with different query shapes
    // 1. Test with an Aabb that contains some spheres
    Aabb queryAabb = Aabb(Real3{-1.0_r, -1.0_r, -1.0_r}, Real3{1.0_r, 1.0_r, 1.0_r});
    TestFindIntersectingElements(tree, object, queryAabb, kNumSpheres);

    // 2. Test with a Sphere as the query shape
    Sphere querySphere = Sphere(Real3{0.0_r, 0.0_r, 0.0_r}, 1.0_r);
    TestFindIntersectingElements(tree, object, querySphere, kNumSpheres);

    // 3. Test with an Obb as the query shape
    Matrix3x3r rotation = Eye<3, real>(); // Identity rotation
    Real3 translation{0.0_r, 0.0_r, 0.0_r};
    MatrixTransformRT transform(rotation, translation);
    Obb queryObb(transform, Real3{1.0_r, 1.0_r, 1.0_r});
    TestFindIntersectingElements(tree, object, queryObb, kNumSpheres);

    // 4. Test with AnyShape
    AnyShape queryShape = Sphere(Real3{0.0_r, 0.0_r, 0.0_r}, 1.0_r);
    TestFindIntersectingElements(tree, object, queryShape, kNumSpheres);
  }
}

TEST(BvhTree, SphereCloud_FindIntersectingElements) {
  SphereCloud_FindIntersectingElements<Aabb>();
  SphereCloud_FindIntersectingElements<Sphere>();
}

/*************************************************************************************************/
/* Test for edge cases in FindIntersectingElements */

TEST(BvhTree, FindIntersectingElements_SdfQueryVisitsShallowLeaves) {
  std::vector<Sphere> spheres;
  spheres.reserve(65);
  // The distant sphere becomes a shallow leaf while the coincident spheres force a deep branch.
  for (int i = 0; i < 64; ++i) {
    spheres.emplace_back(Real3{0_r, 0_r, 0_r}, 0.01_r);
  }
  spheres.emplace_back(Real3{1000_r, 0_r, 0_r}, 0.01_r);

  BvhTreeParams treeParams{};
  treeParams.maxElementsPerLeaf = 1;
  treeParams.maxDepthPerBranch = 1000;
  SphereCloudBvhObject<Sphere> object(spheres);
  BvhTree<Sphere> tree(&object, treeParams);

  // Plane SDF: positive near the deep branch, negative at the distant shallow leaf.
  Int3 const gridSize{2, 2, 2};
  Aabb const gridBounds{Real3{0_r, -1_r, -1_r}, Real3{1000_r, 1_r, 1_r}};
  Aabb const negativeValueBounds{Real3{999_r, -1_r, -1_r}, gridBounds.GetMax()};
  auto grid = std::make_shared<DenseGrid3D<real>>(gridSize, gridBounds, negativeValueBounds);
  for (int x = 0; x < gridSize[0]; ++x) {
    for (int y = 0; y < gridSize[1]; ++y) {
      for (int z = 0; z < gridSize[2]; ++z) {
        Int3 const index{x, y, z};
        (*grid)(index) = 999_r - grid->GetPointOf(index)[0];
      }
    }
  }
  GridSdf const gridSdf{grid, VEye<4>()};
  SdfBv const sdfBv{
      .gridSdf = &gridSdf,
      .distanceThreshold = 0_r,
      .gridFromPointsT = gridSdf.GetGridFromActorTranspose()};

  DynamicArray<int> results;
  tree.FindIntersectingElements(sdfBv, results);

  EXPECT_SPAN_EQ((DynamicArray<int>{64}), results);
}

// Helper function to test edge cases for FindIntersectingElements
template <typename Bv, typename QueryShape>
static void TestFindIntersectingElementsEdgeCase(
    BvhTree<Bv>& tree,
    QueryShape const& queryShape,
    int expectedResultCount,
    bool verifyAllElements = false) {
  DynamicArray<int> results;
  tree.template FindIntersectingElements<false>(queryShape, results);
  EXPECT_EQ(results.size(), expectedResultCount);

  // For the "all elements" case, verify each element is included
  if (verifyAllElements && expectedResultCount > 0) {
    std::vector<bool> found(expectedResultCount, false);
    for (int i = 0; i < isize(results); ++i) {
      EXPECT_GE(results[i], 0);
      EXPECT_LT(results[i], expectedResultCount);
      found[results[i]] = true;
    }

    for (int i = 0; i < expectedResultCount; ++i) {
      EXPECT_TRUE(found[i]);
    }
  }
}

template <typename Bv>
static void SphereCloud_FindIntersectingElements_EdgeCases() {
  auto constexpr kNumSpheres = kSphereCloud_Points.dims[1];

  // Build sphere cloud from data.
  std::vector<Sphere> spheres(kNumSpheres);
  for (int i = 0; i < kNumSpheres; ++i) {
    spheres[i] = Sphere(kSphereCloud_Points[0][i], kSphereCloud_Radius);
  }

  // Build Bvh tree.
  BvhTreeParams treeParams{};
  treeParams.maxElementsPerLeaf = 1;
  treeParams.maxDepthPerBranch = 1000;
  SphereCloudBvhObject<Bv> object(spheres);
  BvhTree<Bv> tree(&object, treeParams);

  // 1. Test with a query that doesn't intersect any elements
  Aabb queryAabb = Aabb(Real3{100.0_r, 100.0_r, 100.0_r}, Real3{101.0_r, 101.0_r, 101.0_r});
  TestFindIntersectingElementsEdgeCase(tree, queryAabb, 0);

  // 2. Test with a query that intersects all elements
  queryAabb = Aabb(Real3{-100.0_r, -100.0_r, -100.0_r}, Real3{100.0_r, 100.0_r, 100.0_r});
  TestFindIntersectingElementsEdgeCase(tree, queryAabb, kNumSpheres, true);
}

TEST(BvhTree, SphereCloud_FindIntersectingElements_EdgeCases) {
  SphereCloud_FindIntersectingElements_EdgeCases<Aabb>();
  SphereCloud_FindIntersectingElements_EdgeCases<Sphere>();
}
