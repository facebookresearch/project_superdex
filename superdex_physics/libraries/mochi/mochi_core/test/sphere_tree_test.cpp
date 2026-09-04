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
#include <mochi_core/geometry/plane.h>
#include <mochi_core/geometry/sphere_tree.h>
#include <mochi_core/mochi_config.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/rand_utils.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>

using namespace mochi;
using namespace mochi::test;

class SphereOctTreeTest : public testing::TestWithParam<int> {};

// Select a cut through the tree at each depth using ForEachNodeSphere. Leaves above the cut
// terminate shorter branches, while non-leaf nodes at the cut represent branches that continue
// below it. The spheres in each cut must collectively cover every input point.
static void
CheckCoverageAtEveryDepth(SphereOctTree const& tree, Span<Real3 const> points, int maxCutDepth) {
  for (int cutDepth = 0; cutDepth <= maxCutDepth; ++cutDepth) {
    DynamicArray<Sphere> coveringSpheres;
    tree.ForEachNodeSphere([&](Sphere const& sphere, int depth, bool isLeaf) {
      if (depth == cutDepth || (isLeaf && depth < cutDepth)) {
        coveringSpheres.push_back(sphere);
      }
    });

    for (int i = 0; i < isize(points); ++i) {
      Real3 const& point = points[i];
      bool covered = false;
      for (Sphere const& sphere : coveringSpheres) {
        Real3 const delta = point - sphere.GetCenter();
        real const distance = std::hypot(delta[0], delta[1], delta[2]);
        if (distance <= sphere.GetRadius()) {
          covered = true;
          break;
        }
      }
      EXPECT_TRUE(covered) << "Point " << i << " (" << point[0] << ", " << point[1] << ", "
                           << point[2] << ") is not covered by the tree cut at depth " << cutDepth;
    }
  }
}

TEST_P(SphereOctTreeTest, FromRandomPoints) {
  int const maxPerLeaf = GetParam();
  auto rng = RandomGenerator(42);
  DynamicArray<Real3> points;
  for (int numPoints = 0; numPoints < 250; ++numPoints) {
    for (real scale : {0.01_r, 1_r, 100_r}) {
      points.resize_noinit(static_cast<size_t>(numPoints));
      SetRandom(rng, -scale, scale, MakeSpan(points));
      auto tree = SphereOctTree::FromPoints(points, maxPerLeaf);
      tree.AssertTreeIsValid(points);
    }
  }
}

TEST_P(SphereOctTreeTest, CoincidentPoints) {
  int const maxPerLeaf = GetParam();
  Real3 const location = {1_r, 2_r, 3_r};
  for (int numPoints = 0; numPoints < 250; ++numPoints) {
    DynamicArray<Real3> points;
    points.resize_noinit(numPoints);
    std::fill(points.begin(), points.end(), location);
    auto tree = SphereOctTree::FromPoints(points, maxPerLeaf);
    tree.AssertTreeIsValid(points);
  }
}

TEST_P(SphereOctTreeTest, FindIntersectingSamplesAll) {
  int const maxPerLeaf = GetParam();
  auto rng = RandomGenerator(123);
  for (int numPoints : {0, 1, 2, 7, 8, 9, 50, 200}) {
    DynamicArray<Real3> points;
    points.resize_noinit(numPoints);
    SetRandom(rng, -10_r, 10_r, MakeSpan(points));
    auto tree = SphereOctTree::FromPoints(points, maxPerLeaf);
    tree.AssertTreeIsValid(points);

    DynamicArray<int> result;
    tree.FindIntersectingSamplesFn(
        [](BatchSphere<8> const&) {
          return VEqual(Vec8r{}, Vec8r{}); // all true
        },
        result);

    // Should return every index exactly once.
    std::sort(result.begin(), result.end());
    ASSERT_EQ(isize(result), numPoints);
    for (int i = 0; i < numPoints; ++i) {
      EXPECT_EQ(result[i], i);
    }
  }
}

TEST_P(SphereOctTreeTest, FindIntersectingSamplesNone) {
  int const maxPerLeaf = GetParam();
  auto rng = RandomGenerator(456);
  for (int numPoints : {0, 1, 2, 7, 8, 9, 50, 200}) {
    DynamicArray<Real3> points;
    points.resize_noinit(numPoints);
    SetRandom(rng, -10_r, 10_r, MakeSpan(points));
    auto tree = SphereOctTree::FromPoints(points, maxPerLeaf);
    tree.AssertTreeIsValid(points);

    DynamicArray<int> result;
    result.push_back(123);
    tree.FindIntersectingSamplesFn(
        [](BatchSphere<8> const&) {
          return Vec8r{0}; // all false
        },
        result);

    EXPECT_EQ(isize(result), 0);
  }
}

TEST(SphereOctTreeTest, FindIntersectingSamplesPrimitiveShape) {
  DynamicArray<int> result{123};

  {
    DynamicArray<Real3> noPoints;
    auto tree = SphereOctTree::FromPoints(noPoints, 1);
    tree.AssertTreeIsValid(noPoints);
    tree.FindIntersectingSamples(Plane{Real3{1_r, 0_r, 0_r}, 0_r}, result);
    EXPECT_TRUE(result.empty());
  }

  {
    DynamicArray<Real3> const points{
        Real3{-1_r, 0_r, 0_r}, Real3{0_r, 0_r, 0_r}, Real3{1_r, 0_r, 0_r}};

    auto const tree = SphereOctTree::FromPoints(points, 1);
    tree.AssertTreeIsValid(points);

    result.push_back(123);
    tree.FindIntersectingSamples(Plane{Real3{1_r, 0_r, 0_r}, -2_r}, result);
    EXPECT_TRUE(result.empty());

    tree.FindIntersectingSamples(Plane{Real3{1_r, 0_r, 0_r}, 0_r}, result);
    std::sort(result.begin(), result.end());
    DynamicArray<int> const expected{0, 1};
    EXPECT_EQ(result, expected);
  }
}

#if MOCHI_USE_DOUBLE_PRECISION
TEST(SphereOctTreeTest, ChildSphereRadiusIncludesEveryCenterRoundingError) {
  real const xError = std::ldexp(1_r, -25);
  real const yError = std::ldexp(1_r, -200);
  DynamicArray<Real3> const points{Real3{1_r + xError, yError, 0_r}};
  auto tree = SphereOctTree::FromPoints(points, 1);
  tree.AssertTreeIsValid(points);

  real leafRadius = -1_r;
  tree.ForEachNodeSphere([&](Sphere const& sphere, int, bool isLeaf) {
    if (isLeaf) {
      leafRadius = sphere.GetRadius();
    }
  });

  // The smaller y error is lost when added to xError in round-to-nearest double arithmetic.
  // Outward rounding must nevertheless make the stored radius larger than xError.
  EXPECT_GT(leafRadius, xError);
}
#endif // MOCHI_USE_DOUBLE_PRECISION

TEST(SphereOctTreeTest, MovePreservesSimdIndexPadding) {
  DynamicArray<Real3> const points{Real3{1_r, 2_r, 3_r}};
  auto original = SphereOctTree::FromPoints(points, std::numeric_limits<int>::max());
  auto moveConstructed = std::move(original);

  DynamicArray<Real3> const noPoints;
  auto moveAssigned = SphereOctTree::FromPoints(noPoints, 1);
  moveAssigned = std::move(moveConstructed);

  DynamicArray<int> result;
  moveAssigned.FindIntersectingSamplesFn(
      [](BatchSphere<8> const&) { return VEqual(Vec8r{}, Vec8r{}); }, result);

  DynamicArray<int> const expected{0};
  EXPECT_EQ(result, expected);
}

TEST_P(SphereOctTreeTest, FindIntersectingSamplesPartial) {
  int const maxPerLeaf = GetParam();
  auto rng = RandomGenerator(789);
  for (int numPoints : {1, 2, 7, 8, 9, 50, 200}) {
    DynamicArray<Real3> points;
    points.resize_noinit(numPoints);
    SetRandom(rng, -10_r, 10_r, MakeSpan(points));
    auto tree = SphereOctTree::FromPoints(points, maxPerLeaf);
    tree.AssertTreeIsValid(points);

    // Sweep the half-space plane x <= threshold across many threshold values.
    int const numSteps = 20;
    for (int step = 0; step <= numSteps; ++step) {
      real const threshold = Lerp(-12_r, 12_r, static_cast<real>(step) / numSteps);

      // A sphere overlaps the half-space (x <= threshold) if center.x - radius <= threshold.
      DynamicArray<int> result;
      tree.FindIntersectingSamplesFn(
          [threshold](BatchSphere<8> const& spheres) {
            return (spheres.center[0] - spheres.radius) <= Vec8r{threshold};
          },
          result);

      // Mark which points were returned by FindIntersectingSamples.
      DynamicArray<bool> found(numPoints, false);
      for (int idx : result) {
        ASSERT_GE(idx, 0);
        ASSERT_LT(idx, numPoints);
        EXPECT_FALSE(found[idx]) << "Duplicate index " << idx << " in result";
        found[idx] = true;
      }

      // Every point NOT in the result must be in front of the plane (x > threshold).
      // This verifies there are no false negatives.
      for (int i = 0; i < numPoints; ++i) {
        if (!found[i]) {
          EXPECT_GT(points[i][0], threshold)
              << "Point " << i << " (x=" << points[i][0]
              << ") is behind the plane at threshold=" << threshold << " but was not found";
        }
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    SphereOctTree,
    SphereOctTreeTest,
    testing::Values(1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 500));

TEST(SphereOctTreeTest, ForEachNodeSphereUsesBreadthFirstConstructionOrder) {
  int constexpr kNumPoints = 65;
  DynamicArray<Real3> points;
  points.resize_noinit(kNumPoints);
  for (int i = 0; i < kNumPoints; ++i) {
    points[i] = Real3{static_cast<real>(i), 0_r, 0_r};
  }
  auto tree = SphereOctTree::FromPoints(points, 1);
  tree.AssertTreeIsValid(points);

  struct Visit {
    real centerX;
    int depth;
    bool isLeaf;
  };
  DynamicArray<Visit> visits;
  tree.ForEachNodeSphere([&](Sphere const& sphere, int depth, bool isLeaf) {
    visits.push_back({sphere.GetCenter()[0], depth, isLeaf});
  });

  std::array<int, 4> internalCounts{};
  std::array<int, 4> leafCounts{};
  int previousDepth = -1;
  real previousCenterX = std::numeric_limits<real>::lowest();
  for (Visit const& visit : visits) {
    ASSERT_GE(visit.depth, previousDepth);
    ASSERT_GE(visit.depth, 0);
    ASSERT_LT(static_cast<size_t>(visit.depth), internalCounts.size());
    if (visit.depth != previousDepth) {
      previousDepth = visit.depth;
      previousCenterX = std::numeric_limits<real>::lowest();
    }
    EXPECT_GE(visit.centerX, previousCenterX);
    previousCenterX = visit.centerX;
    std::array<int, 4>& counts = visit.isLeaf ? leafCounts : internalCounts;
    ++counts[static_cast<size_t>(visit.depth)];
  }

  std::array<int, 4> const expectedInternalCounts{1, 8, 1, 0};
  std::array<int, 4> const expectedLeafCounts{0, 0, 63, 2};
  EXPECT_EQ(internalCounts, expectedInternalCounts);
  EXPECT_EQ(leafCounts, expectedLeafCounts);
  CheckCoverageAtEveryDepth(tree, points, 3);
}
