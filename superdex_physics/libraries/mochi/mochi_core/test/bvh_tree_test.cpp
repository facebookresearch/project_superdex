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
#include <mochi_core/geometry/aabb.h>
#include <mochi_core/geometry/bvh_tree.h>
#include <mochi_core/geometry/geometry_utils.h>
#include <mochi_core/geometry/tetrahedral_mesh.h>
#include <mochi_core/geometry/triangular_mesh.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/dynamic_array.h>
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
/* Nearest-element queries on real meshes, compared against brute force */

// In single precision, the signed box distance (VDistancePointShapeSqr) can round above zero for a
// point exactly on a box boundary and prune the box that contains it. The resulting distance error
// is ~1e-9 m; these tests allow 1e-6 m.
constexpr real kSurfacePruneTol = 1e-6_r; // [m]

// Normals of different but equally close faces agree within this; most agree bitwise.
constexpr real kNormalCosTol = 1e-4_r;

static DynamicArray<Real3> CreateMeshQueryPoints(TriangularMesh const& mesh) {
  constexpr int kPointsPerAxis = 6;
  constexpr int kSurfaceStride = 8;
  DynamicArray<Real3> points;
  Aabb const bounds = ExpandShape(mesh.GetAabb(), 0.2_r * Max(mesh.GetAabb().GetSize()));
  Real3 const step = bounds.GetSize() / static_cast<real>(kPointsPerAxis - 1);
  for (int x = 0; x < kPointsPerAxis; ++x) {
    for (int y = 0; y < kPointsPerAxis; ++y) {
      for (int z = 0; z < kPointsPerAxis; ++z) {
        points.push_back(bounds.GetMin() + StaticCast<Real3>(Int3{x, y, z}) * step);
      }
    }
  }
  auto const coords = mesh.GetNodeCoordinates();
  auto const faces = mesh.GetElementConnectivity();
  for (int f = 0; f < isize(faces); f += kSurfaceStride) {
    points.push_back((coords[faces[f][0]] + coords[faces[f][1]] + coords[faces[f][2]]) / 3_r);
  }
  for (int n = 0; n < isize(coords); n += kSurfaceStride) {
    points.push_back(coords[n]);
    points.push_back(coords[n] + Real3{1e-3_r, -2e-3_r, 3e-3_r});
  }
  return points;
}

// The tree must find an element as close as a brute-force scan finds, up to kSurfacePruneTol. Both
// use the object's own element distance, so without that pruning the distances match bitwise.
static void ExpectFindClosestMatchesBruteForce(
    char const* treeName,
    AabbTree const& tree,
    Span<Real3 const> points) {
  SCOPED_TRACE(treeName);
  BvhObject<Aabb> const& object = *tree.GetObject();
  int numTooFar = 0;
  int numInconsistent = 0;
  for (auto const& p : points) {
    Vec4r const q = ToSimd(p);
    real bruteDistSqr = std::numeric_limits<real>::infinity();
    for (int e = 0; e < object.GetNumElements(); ++e) {
      bruteDistSqr = Min(bruteDistSqr, Get0(object.VGetDistanceSqr(q, e)));
    }
    real distSqr = {};
    int const index = tree.VFindClosest(q, &distSqr);
    numTooFar += std::sqrt(distSqr) <= std::sqrt(bruteDistSqr) + kSurfacePruneTol ? 0 : 1;
    // The returned distance belongs to the returned element.
    numInconsistent += Get0(object.VGetDistanceSqr(q, index)) == distSqr ? 0 : 1;
  }
  EXPECT_EQ(numTooFar, 0) << "out of " << points.size() << " points";
  EXPECT_EQ(numInconsistent, 0) << "out of " << points.size() << " points";
}

TEST(BvhTree, FindClosest_Stats) {
  // Four points on a line, one per leaf: the root splits {0, 1} | {2, 3}.
  DynamicArray<Real3> const points = {
      {0_r, 0_r, 0_r}, {1_r, 0_r, 0_r}, {2_r, 0_r, 0_r}, {3_r, 0_r, 0_r}};
  PointSetAabbObject const object(points);
  AabbTree const tree(
      &object,
      BvhTreeParams{
          .splittingAlgorithm = BvhSplittingAlgorithm::TopDown_Mean, .maxElementsPerLeaf = 1});

  struct Case {
    real x = 0_r;
    int expectedIndex = 0;
    BvhNearestQueryStats expectedStats;
  };
  Case const cases[] = {
      // The root box is no closer than the seed (point 0), so nothing is visited.
      {.x = -10_r,
       .expectedIndex = 0,
       .expectedStats = {.nodesVisited = 0, .bvTests = 1, .elementTests = 1}},
      // Root, then {0, 1}. Both leaves are no closer than the seed.
      {.x = 0.4_r,
       .expectedIndex = 0,
       .expectedStats = {.nodesVisited = 2, .bvTests = 5, .elementTests = 1}},
      // Root, {0, 1} and its leaf {1} (leaf {0} only ties the seed), then {2, 3} and both its
      // leaves.
      {.x = 2.6_r,
       .expectedIndex = 3,
       .expectedStats = {.nodesVisited = 6, .bvTests = 7, .elementTests = 4}},
  };
  for (auto const& c : cases) {
    SCOPED_TRACE(testing::Message() << "x = " << c.x);
    Vec4r const q = ToSimd(Real3{c.x, 0_r, 0_r});
    real distSqr = {};
    BvhNearestQueryStats stats;
    EXPECT_EQ(tree.FindClosestWithStats(q, stats, &distSqr), c.expectedIndex);
    real plainDistSqr = {};
    EXPECT_EQ(tree.VFindClosest(q, &plainDistSqr), c.expectedIndex);
    EXPECT_EQ(distSqr, plainDistSqr);
    EXPECT_EQ(stats.nodesVisited, c.expectedStats.nodesVisited);
    EXPECT_EQ(stats.bvTests, c.expectedStats.bvTests);
    EXPECT_EQ(stats.elementTests, c.expectedStats.elementTests);

    // Stats accumulate across calls.
    tree.FindClosestWithStats(q, stats);
    EXPECT_EQ(stats.elementTests, 2 * c.expectedStats.elementTests);
  }
}

TEST_IF(MOCHI_INTERNAL, BvhTree, FindClosest_MatchesBruteForceOnMesh) {
  std::shared_ptr<TetrahedralMesh const> const tetMesh =
      LoadTetrahedralMesh(test::GetAssetPath("duck/duck_coarse_mesh.mochi.json"), test::ExpectOK{});
  auto const triMesh = std::make_shared<TriangularMesh const>(CreateBoundaryMesh(*tetMesh));
  auto const points = CreateMeshQueryPoints(*triMesh);

  // Providers read these coordinates, so deforming them and refitting changes the trees.
  DynamicArray<Real3> triCoords(triMesh->GetNodeCoordinates());
  DynamicArray<Real3> tetCoords(tetMesh->GetNodeCoordinates());

  TriangularMeshBvhObject<Aabb> const triObject(triMesh, triCoords);
  TetrahedralMeshAabbObject const tetObject(tetMesh, tetCoords);
  PointSetAabbObject const pointObject(tetCoords);
  AabbTree triTree(&triObject, BvhTreeParams{});
  AabbTree tetTree(&tetObject, BvhTreeParams{});
  AabbTree pointTree(
      &pointObject, BvhTreeParams{.splittingAlgorithm = BvhSplittingAlgorithm::TopDown_Mean});

  auto const expectAllMatch = [&]() {
    ExpectFindClosestMatchesBruteForce("Triangle tree", triTree, points);
    ExpectFindClosestMatchesBruteForce("Tet tree", tetTree, points);
    ExpectFindClosestMatchesBruteForce("Point tree", pointTree, points);
  };
  {
    SCOPED_TRACE("Fresh trees");
    expectAllMatch();
  }

  // Deform anisotropically and refit.
  auto const deform = [](Real3 const& x) {
    return Real3{1.2_r * x[0], 0.8_r * x[1] + 0.1_r * std::sin(5_r * x[0]), x[2]};
  };
  std::ranges::transform(triCoords, triCoords.begin(), deform);
  std::ranges::transform(tetCoords, tetCoords.begin(), deform);
  triTree.Refit();
  tetTree.Refit();
  pointTree.Refit();
  {
    SCOPED_TRACE("Refit trees");
    expectAllMatch();
  }
}

// MeshCollider must return the same signed distance, up to kSurfacePruneTol, and an equivalent
// normal with and without the tree, even when the two searches pick different but equally close
// faces.
static void ExpectMeshColliderMatchesBruteForce(char const* meshPath) {
  SCOPED_TRACE(meshPath);
  auto const tetMesh = LoadTetrahedralMesh(test::GetAssetPath(meshPath), test::ExpectOK{});
  auto const triMesh = std::make_shared<TriangularMesh const>(CreateBoundaryMesh(*tetMesh));
  MeshCollider collider(triMesh);
  collider.Initialize();
  ASSERT_TRUE(collider.IsMeshClosed());

  ContactDetectionParams accelerated;
  accelerated.tolerance = std::numeric_limits<real>::infinity();
  ContactDetectionParams bruteForce = accelerated;
  bruteForce.useAccelerationStructures = false;

  int numSdfMismatches = 0;
  int numNormalMismatches = 0;
  auto const points = CreateMeshQueryPoints(*triMesh);
  for (auto const& p : points) {
    Vec4r pos, grad, bruteForcePos, bruteForceGrad;
    real sdf = {}, bruteForceSdf = {};
    ASSERT_TRUE(collider.QueryPoint(ToSimd(p, 1_r), accelerated, pos, sdf, grad));
    ASSERT_TRUE(collider.QueryPoint(
        ToSimd(p, 1_r), bruteForce, bruteForcePos, bruteForceSdf, bruteForceGrad));
    numSdfMismatches += std::abs(sdf - bruteForceSdf) <= kSurfacePruneTol ? 0 : 1;
    numNormalMismatches += Dot<3>(grad, bruteForceGrad) > 1_r - kNormalCosTol ? 0 : 1;
  }
  EXPECT_EQ(numSdfMismatches, 0) << "out of " << points.size() << " points";
  EXPECT_EQ(numNormalMismatches, 0) << "out of " << points.size() << " points";
}

TEST_IF(MOCHI_INTERNAL, BvhTree, MeshCollider_MatchesBruteForceOnMeshes) {
  ExpectMeshColliderMatchesBruteForce("duck/duck_coarse_mesh.mochi.json");
  ExpectMeshColliderMatchesBruteForce("armadillo/armadillo_coarse_mesh.mochi.json");
}
