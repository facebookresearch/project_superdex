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

#include <mochi_core/geometry/ray.h>

#include <mochi_core/test/mochi_test_helpers.h>

#include <algorithm>
#include <array>
#include <iterator>
#include <random>

using namespace mochi;

// Coordinates and connectivity for a box
constexpr std::array<Real3, 8> const kCoordinates = {
    Real3{-1.0_r, -1.0_r, -1.0_r}, // 0
    Real3{1.0_r, -1.0_r, -1.0_r}, // 1
    Real3{-1.0_r, 1.0_r, -1.0_r}, // 2
    Real3{1.0_r, 1.0_r, -1.0_r}, // 3
    Real3{-1.0_r, -1.0_r, 1.0_r}, // 4
    Real3{1.0_r, -1.0_r, 1.0_r}, // 5
    Real3{-1.0_r, 1.0_r, 1.0_r}, // 6
    Real3{1.0_r, 1.0_r, 1.0_r}, // 7
};
constexpr std::array<Int3, 12> const kConnectivity = {
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

static TriangleSoup BoxTriangleSoup() {
  TriangleSoup soup;
  std::transform(
      kConnectivity.begin(),
      kConnectivity.end(),
      std::back_inserter(soup.triangles),
      [&](Int3 const& idx) {
        return Triangle{kCoordinates[idx[0]], kCoordinates[idx[1]], kCoordinates[idx[2]]};
      });
  return soup;
}

TEST(Ray, RayTriangleIntersection) {
  auto tri =
      Triangle{Real3{0.0_r, 0.0_r, 0.0_r}, Real3{1.0_r, 0.0_r, 0.0_r}, Real3{0.0_r, 1.0_r, 0.0_r}};

  auto ray1 = Ray{Real3{0.2_r, 0.2_r, 1.0_r}, Real3{0.0_r, 0.0_r, -1.0_r}};
  auto ray2 = Ray{Real3{0.0_r, 0.0_r, 1.0_r}, Real3{1.0_r, 0.0_r, 0.0_r}};

  auto rayResult1 = RayCast(ray1, tri);
  auto rayResult2 = RayCast(ray2, tri);

  EXPECT_TRUE(rayResult1.has_value());
  EXPECT_NEAR_EQ(rayResult1->t, 1.0_r);
  EXPECT_NEAR_EQ(ToReal3(rayResult1->intersection), (Real3{0.2_r, 0.2_r, 0.0_r}));

  EXPECT_FALSE(rayResult2.has_value());
}

TEST(Ray, RayBoxIntersection) {
  // Ray origin outside the box
  auto aabb = Aabb{Real3{-1.0_r, -1.0_r, -1.0_r}, Real3{1.0_r, 1.0_r, 1.0_r}};

  auto ray1 = Ray{Real3{-2.0_r, -2.0_r, -2.0_r}, Real3{1.0_r, 1.0_r, 1.0_r}};
  auto ray2 = Ray{Real3{-2.0_r, 0.0_r, 0.0_r}, Real3{1.0_r, 0.0_r, 0.0_r}};
  auto ray3 = Ray{Real3{-2.0_r, 0.0_r, 0.0_r}, Real3{0.0_r, 1.0_r, 0.0_r}};

  auto rayResult1 = RayCast(ray1, aabb);
  auto rayResult2 = RayCast(ray2, aabb);
  auto rayResult3 = RayCast(ray3, aabb);

  ASSERT_TRUE(rayResult1.has_value());
  EXPECT_NEAR_EQ(rayResult1->t, 1.0_r);
  EXPECT_NEAR_EQ(ToReal3(rayResult1->intersection), (Real3{-1.0_r, -1.0_r, -1.0_r}));

  ASSERT_TRUE(rayResult2.has_value());
  EXPECT_NEAR_EQ(rayResult2->t, 1.0_r);
  EXPECT_NEAR_EQ(ToReal3(rayResult2->intersection), (Real3{-1.0_r, 0.0_r, 0.0_r}));

  EXPECT_FALSE(rayResult3.has_value());

  // Ray origin inside the box
  auto ray4 = Ray{Real3{0.0_r, 0.0_r, 0.0_r}, Real3{1.0_r, 0.0_r, 0.0_r}};
  EXPECT_TRUE(RayCast(ray4, aabb).has_value());

  auto ray5 = Ray{Real3{0.0_r, 0.0_r, 0.0_r}, Real3{0.0_r, 0.0_r, -1.0_r}};
  EXPECT_TRUE(RayCast(ray5, aabb).has_value());

  auto ray6 = Ray{Real3{0.5_r, -0.5_r, 0.3_r}, Normalize(Real3{1.0_r, 1.0_r, 1.0_r})};
  EXPECT_TRUE(RayCast(ray6, aabb).has_value());

  // Ray origin on the surface of the box
  auto ray7 = Ray{Real3{-1.0_r, 0.0_r, 0.0_r}, Real3{1.0_r, 0.0_r, 0.0_r}};
  auto result7 = RayCast(ray7, aabb);
  ASSERT_TRUE(result7.has_value());
  EXPECT_NEAR_EQ(result7->t, 0.0_r);

  auto ray8 = Ray{Real3{1.0_r, 1.0_r, 1.0_r}, Normalize(Real3{-1.0_r, -1.0_r, -1.0_r})};
  auto result8 = RayCast(ray8, aabb);
  ASSERT_TRUE(result8.has_value());
  EXPECT_NEAR_EQ(result8->t, 0.0_r);
}

TEST(Ray, RayTriangleSoupIntersection) {
  // Test against three triangles
  {
    auto tri1 = Triangle{
        Real3{0.0_r, 0.0_r, -2.0_r}, Real3{1.0_r, 0.0_r, -2.0_r}, Real3{0.0_r, 1.0_r, -2.0_r}};
    auto tri2 = Triangle{
        Real3{0.0_r, 0.0_r, -1.0_r}, Real3{1.0_r, 0.0_r, -1.0_r}, Real3{0.0_r, 1.0_r, -1.0_r}};
    auto tri3 = Triangle{
        Real3{0.0_r, 0.0_r, 0.0_r}, Real3{1.0_r, 0.0_r, 0.0_r}, Real3{0.0_r, 1.0_r, 0.0_r}};

    auto triangleSoup = TriangleSoup{{tri1, tri2, tri3}};

    auto ray1 = Ray{Real3{0.2_r, 0.2_r, 1.0_r}, Real3{0.0_r, 0.0_r, -1.0_r}};
    auto ray2 = Ray{Real3{0.0_r, 0.0_r, 1.0_r}, Real3{1.0_r, 0.0_r, 0.0_r}};

    auto rayResult1 = RayCast(ray1, triangleSoup);
    auto rayResult2 = RayCast(ray2, triangleSoup);

    EXPECT_TRUE(rayResult1.has_value());
    EXPECT_NEAR_EQ(rayResult1->t, 1.0_r);
    EXPECT_NEAR_EQ(ToReal3(rayResult1->intersection), (Real3{0.2_r, 0.2_r, 0.0_r}));

    EXPECT_FALSE(rayResult2.has_value());
  }

  // Test against a box
  {
    auto boxTriangleSoup = BoxTriangleSoup();

    auto ray1 = Ray{Real3{-2.0_r, -2.0_r, -2.0_r}, Real3{1.0_r, 1.0_r, 1.0_r}};
    auto ray2 = Ray{Real3{-2.0_r, 0.0_r, 0.0_r}, Real3{1.0_r, 0.0_r, 0.0_r}};
    auto ray3 = Ray{Real3{-2.0_r, 0.0_r, 0.0_r}, Real3{0.0_r, 1.0_r, 0.0_r}};

    auto rayResult1 = RayCast(ray1, boxTriangleSoup);
    auto rayResult2 = RayCast(ray2, boxTriangleSoup);
    auto rayResult3 = RayCast(ray3, boxTriangleSoup);

    EXPECT_TRUE(rayResult1.has_value());
    EXPECT_NEAR_EQ(rayResult1->t, 1.0_r);
    EXPECT_NEAR_EQ(ToReal3(rayResult1->intersection), (Real3{-1.0_r, -1.0_r, -1.0_r}));

    EXPECT_TRUE(rayResult2.has_value());
    EXPECT_NEAR_EQ(rayResult2->t, 1.0_r);
    EXPECT_NEAR_EQ(ToReal3(rayResult2->intersection), (Real3{-1.0_r, 0.0_r, 0.0_r}));

    EXPECT_FALSE(rayResult3.has_value());
  }
}

TEST(Ray, RayBvhIntersection) {
  int constexpr kKumTests = 100;
  auto boxTriangleSoup = BoxTriangleSoup();
  std::default_random_engine generator(100);

  BvhTreeParams params;
  params.maxElementsPerLeaf = 2;
  AabbTree tree(&boxTriangleSoup, params);

  // Random rays from outside the box — compare BVH vs brute force
  std::normal_distribution<real> directionDist(0.0, 1.0);
  auto generateRandomUnitVector = [&]() {
    Real3 v{directionDist(generator), directionDist(generator), directionDist(generator)};
    return Normalize(v);
  };

  for (int i = 0; i < kKumTests; ++i) {
    auto v = 10.0_r * generateRandomUnitVector();

    Ray ray{v, -v};
    auto bvhRay = RayCast(ray, tree, boxTriangleSoup);
    auto bruteForceRay = RayCast(ray, boxTriangleSoup);

    EXPECT_TRUE(bvhRay.has_value() == bruteForceRay.has_value());
    if (bvhRay && bruteForceRay) {
      EXPECT_NEAR_EQ(bvhRay->t, bruteForceRay->t);
      EXPECT_NEAR_EQ(bvhRay->intersection, bruteForceRay->intersection);
      EXPECT_EQ(bvhRay->index, bruteForceRay->index);
    }
  }

  // Origin at center of the box, shooting along +x
  {
    auto ray = Ray{Real3{0.0_r, 0.0_r, 0.0_r}, Real3{1.0_r, 0.0_r, 0.0_r}};
    auto bvhResult = RayCast(ray, tree, boxTriangleSoup);
    auto bruteForceResult = RayCast(ray, boxTriangleSoup);

    ASSERT_TRUE(bruteForceResult.has_value());
    ASSERT_TRUE(bvhResult.has_value());
    EXPECT_NEAR_EQ(bvhResult->t, bruteForceResult->t);
    EXPECT_NEAR_EQ(bvhResult->intersection, bruteForceResult->intersection);
    EXPECT_EQ(bvhResult->index, bruteForceResult->index);
  }

  // Random rays from inside the box — compare BVH vs brute force
  std::uniform_real_distribution<real> positionDist(-0.9_r, 0.9_r);
  for (int i = 0; i < kKumTests; ++i) {
    Real3 origin{positionDist(generator), positionDist(generator), positionDist(generator)};
    Real3 direction = Normalize(
        Real3{directionDist(generator), directionDist(generator), directionDist(generator)});

    Ray ray{origin, direction};
    auto bvhResult = RayCast(ray, tree, boxTriangleSoup);
    auto bruteForceResult = RayCast(ray, boxTriangleSoup);

    EXPECT_EQ(bvhResult.has_value(), bruteForceResult.has_value());
    if (bvhResult && bruteForceResult) {
      EXPECT_NEAR_EQ(bvhResult->t, bruteForceResult->t);
      EXPECT_NEAR_EQ(bvhResult->intersection, bruteForceResult->intersection);
      EXPECT_EQ(bvhResult->index, bruteForceResult->index);
    }
  }
}
