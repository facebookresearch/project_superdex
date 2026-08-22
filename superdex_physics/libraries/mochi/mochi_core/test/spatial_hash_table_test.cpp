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
#include <mochi_core/geometry/spatial_hash_table.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/rand_utils.h>

using namespace mochi;
using namespace mochi::test;

class SpatialHashTableTest : public ::testing::Test {
 protected:
  static DynamicArray<Real3>
  GenerateRandomPoints(unsigned int seed, int numPoints, real minCoordinate, real maxCoordinate) {
    mochi_default_random_engine randomGenerator = RandomGenerator(seed);
    DynamicArray<Real3> points(numPoints);
    SetRandom(randomGenerator, minCoordinate, maxCoordinate, MakeSpan(points));
    return points;
  }

  static void CompareHashingAndBruteForce(
      SpatialHashTable& spatialHashTable,
      DynamicArray<Real3> const& points) {
    int const numPoints = spatialHashTable.GetCapacity();
    real const cellSize = spatialHashTable.GetCellSize();

    // This is enforcing the validity of the test case calling this function, rather than the
    // correctness of the hash table data structure.
    EXPECT_EQ(numPoints, isize(points));

    spatialHashTable.Reset();
    EXPECT_TRUE(spatialHashTable.GetNumPoints() == 0);

    for (int pointIndex = 0; pointIndex < numPoints; pointIndex++) {
      spatialHashTable.AddPoint(pointIndex, points[pointIndex]);
    }
    EXPECT_TRUE(spatialHashTable.GetNumPoints() == numPoints);
    for (int pointIndex = 0; pointIndex < numPoints; pointIndex++) {
      // Count points within cellSize of the current point using the spatial hash table.
      int numPointsWithinCellSize = 0;
      Real3 const& currentPoint = points[pointIndex];
      std::set<int> nearbyPointIndices;
      spatialHashTable.IteratePointsNearPosition(currentPoint, [&](int nearbyPointIndex) {
        // Nearby point iteration should never repeat the same point.
        EXPECT_TRUE(!nearbyPointIndices.contains(nearbyPointIndex));
        nearbyPointIndices.insert(nearbyPointIndex);
        Real3 const& nearbyPoint = points[nearbyPointIndex];
        if (Norm(nearbyPoint - currentPoint) < cellSize) {
          numPointsWithinCellSize++;
        }
      });

      // Count again via brute-force search of all points.
      int numPointsWithinCellSizeBruteForce = 0;
      for (int otherPointIndex = 0; otherPointIndex < numPoints; otherPointIndex++) {
        Real3 const& otherPoint = points[otherPointIndex];
        if (Norm(otherPoint - currentPoint) < cellSize) {
          numPointsWithinCellSizeBruteForce++;
        }
      }

      // Verify that the spatial hash table has the same results as a brue-force O(N^2) search for
      // nearby points.
      EXPECT_EQ(numPointsWithinCellSizeBruteForce, numPointsWithinCellSize);

      // A point will always be within cell size of itself.
      EXPECT_TRUE(numPointsWithinCellSize > 0);
    } // loop over points
  }
};

// Loop over multiple sets of random points to emulate expected usage within time steping.
TEST_F(SpatialHashTableTest, RefillingWithRandomPoints) {
  int const numPoints = 123;
  int const minNumBins = 456;
  real const cellSize = 2.345_r;
  real const minCoordinate = -4.321_r;
  real const maxCoordinate = 5.678_r;

  // Construct a single spatial hash table with a fixed capacity and minimum number of bins. The
  // requested bin count is intentionally not a power of two; construction rounds it internally.
  SpatialHashTable spatialHashTable(cellSize, numPoints, minNumBins);

  // Repeatedly reset/repopulate the hash table with different sets of points generated from
  // different random seeds, which reflects expected usage for contact queries.
  for (unsigned int seed = 0; seed < 8; seed++) {
    CompareHashingAndBruteForce(
        spatialHashTable, GenerateRandomPoints(seed, numPoints, minCoordinate, maxCoordinate));
  } // loop over random seeds
} // test

// Specifically test a high-load-factor case where there are guaranteed hash collisions.
TEST_F(SpatialHashTableTest, HighLoadFactor) {
  int const numPoints = 456;
  // A number of bins less than 27 guarantees that at least one will be visited multiple times
  // during nearby point iteration.
  int const minNumBins = 16;
  real const cellSize = 2.345_r;
  real const minCoordinate = -1.7_r * cellSize;
  real const maxCoordinate = 1.7_r * cellSize;
  unsigned int const seed = 0;
  SpatialHashTable spatialHashTable(cellSize, numPoints, minNumBins);
  CompareHashingAndBruteForce(
      spatialHashTable, GenerateRandomPoints(seed, numPoints, minCoordinate, maxCoordinate));
}

// IteratePointsNearPosition de-duplicates the 3x3x3 stencil's bins with a lossy pre-filter keyed on
// the low 6 bits of the bin index. With more than 64 bins, two distinct bins can share those low
// bits, exercising the pre-filter's false-positive fall-through (with <= 64 bins, see
// HighLoadFactor, that path is unreachable). CompareHashingAndBruteForce verifies no point is
// dropped or double-reported.
TEST_F(SpatialHashTableTest, PrefilterLowBitCollisions) {
  int const numPoints = 512;
  int const minNumBins = 256; // > 64 so distinct bins can alias on the pre-filter's low 6 bits.
  real const cellSize = 2.345_r;
  real const minCoordinate = -1.7_r * cellSize;
  real const maxCoordinate = 1.7_r * cellSize;
  unsigned int const seed = 0;
  SpatialHashTable spatialHashTable(cellSize, numPoints, minNumBins);
  CompareHashingAndBruteForce(
      spatialHashTable, GenerateRandomPoints(seed, numPoints, minCoordinate, maxCoordinate));
}

TEST_F(SpatialHashTableTest, OccupiedPointBoundsTrackRefills) {
  SpatialHashTable spatialHashTable(1_r, 3, 4);
  spatialHashTable.AddPoint(0, Real3{0.25_r, 1.25_r, -2.25_r});
  spatialHashTable.AddPoint(1, Real3{3.25_r, -4.25_r, 5.25_r});

  Aabb occupiedPointBounds = spatialHashTable.GetOccupiedPointBounds();
  EXPECT_NEAR_EQ((Real3{0.25_r, -4.25_r, -2.25_r}), occupiedPointBounds.GetMin());
  EXPECT_NEAR_EQ((Real3{3.25_r, 1.25_r, 5.25_r}), occupiedPointBounds.GetMax());

  spatialHashTable.Reset();
  EXPECT_EQ(0, spatialHashTable.GetNumPoints());
  spatialHashTable.AddPoint(2, Real3{-7.25_r, 8.25_r, 9.25_r});

  occupiedPointBounds = spatialHashTable.GetOccupiedPointBounds();
  EXPECT_NEAR_EQ((Real3{-7.25_r, 8.25_r, 9.25_r}), occupiedPointBounds.GetMin());
  EXPECT_NEAR_EQ((Real3{-7.25_r, 8.25_r, 9.25_r}), occupiedPointBounds.GetMax());
}

// Sweep through different point distributions, from all in one bin, to all in separate bins, by
// progressively scaling up a stencil of points.
TEST_F(SpatialHashTableTest, DensitySweep) {
  // Form a cluster of points in the unit cube.
  DynamicArray<Real3> points;
  int const n = 3;
  auto const getCoord = [](int i) { return ((real)i + 0.5_r) / ((real)n); };
  for (int i = 0; i < n; i++) {
    real const x = getCoord(i);
    for (int j = 0; j < n; j++) {
      real const y = getCoord(j);
      for (int k = 0; k < n; k++) {
        real const z = getCoord(k);
        points.emplace_back(x, y, z);
      }
    }
  }

  // Set up a hash table with unit cell size, such that all points are initially in a single cell.
  real constexpr kUnitCellSize = 1_r;
  int const numPoints = isize(points);
  int const minNumBins = 128 * numPoints; // Low load factor, collisions checked in other tests.
  SpatialHashTable spatialHashTable(kUnitCellSize, numPoints, minNumBins);

  // Progressively scale up the points until they are all guaranteed to be in separate cells,
  // checking for correctness at each scale.
  int constexpr kNumLevels = 16;
  real constexpr kLevelScale = 1.25_r;
  for (int level = 0; level < kNumLevels; level++) {
    // Check correcntess
    CompareHashingAndBruteForce(spatialHashTable, points);
    // Scale up points
    for (Real3& point : points) {
      point = kLevelScale * point;
    }
  }
}
