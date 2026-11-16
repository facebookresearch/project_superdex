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

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_physics/mochi_physics.h>
#include <superdex_robotics/sensors/tactile_observation.h>

#include <array>
#include <limits>
#include <memory>
#include <string>

using namespace mochi;
using namespace superdex::robotics;

TEST(TactileObservationTest, ScatterProducesDenseForcesAndAccumulatesDuplicates) {
  DynamicArray<Real3> const positions = {{0_r, 0_r, 0_r}, {1_r, 0_r, 0_r}, {2_r, 0_r, 0_r}};
  Error error;
  TactileTaxelLayout const layout = MakeTactileTaxelLayout(positions, error);
  EXPECT_OK(error);
  DynamicArray<NodeContactForce> const sparse = {
      {.index = 1, .force = {1_r, 2_r, 3_r}},
      {.index = 1, .force = {4_r, 5_r, 6_r}},
      {.index = 2, .force = {-1_r, 0_r, 1_r}}};

  TactileTaxelForces const result = ScatterTactileTaxelForces(sparse, TransformRT{}, layout, error);
  EXPECT_OK(error);
  ASSERT_EQ(result.forces.size(), 3);
  EXPECT_EQ(result.forces[0], Real3{});
  EXPECT_EQ(result.forces[1], (Real3{5_r, 7_r, 9_r}));
  EXPECT_EQ(result.forces[2], (Real3{-1_r, 0_r, 1_r}));
}

TEST(TactileObservationTest, ScatterReusesAPreallocatedObservation) {
  DynamicArray<Real3> const positions = {{0_r, 0_r, 0_r}, {1_r, 0_r, 0_r}};
  Error error;
  TactileTaxelLayout const layout = MakeTactileTaxelLayout(positions, error);
  EXPECT_OK(error);
  DynamicArray<NodeContactForce> const sparse = {{.index = 1, .force = {1_r, 2_r, 3_r}}};
  TactileTaxelForces result;
  result.forces.resize(2);
  Real3* const allocation = result.forces.data();

  ScatterTactileTaxelForcesInto(sparse, TransformRT{}, layout, result, error);

  EXPECT_OK(error);
  EXPECT_EQ(result.forces.data(), allocation);
  EXPECT_EQ(result.forces[0], Real3{});
  EXPECT_EQ(result.forces[1], (Real3{1_r, 2_r, 3_r}));
}

TEST(TactileObservationTest, FlattensForcesIntoSharedVertexMajorOrder) {
  TactileTaxelForces forces;
  forces.forces = {{1_r, 2_r, 3_r}, {4_r, 5_r, 6_r}};
  std::array<float, 6> floatValues{};
  Error error;

  FlattenTactileTaxelForcesInto(forces, Span<float>{floatValues}, error);

  EXPECT_OK(error);
  EXPECT_EQ(floatValues, (std::array<float, 6>{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}));
}

TEST(TactileObservationTest, FlattenRejectsAMismatchedOutputSize) {
  TactileTaxelForces forces;
  forces.forces = {{1_r, 2_r, 3_r}};
  std::array<float, 2> floatValues{};
  Error error;

  FlattenTactileTaxelForcesInto(forces, Span<float>{floatValues}, error);

  EXPECT_NOT_OK(error);
}

TEST(TactileObservationTest, ScatterRejectsOutOfRangeTaxelIndex) {
  DynamicArray<Real3> const positions = {{0_r, 0_r, 0_r}};
  Error error;
  TactileTaxelLayout const layout = MakeTactileTaxelLayout(positions, error);
  EXPECT_OK(error);
  DynamicArray<NodeContactForce> const sparse = {{.index = 1, .force = {1_r, 0_r, 0_r}}};

  TactileTaxelForces const result = ScatterTactileTaxelForces(sparse, TransformRT{}, layout, error);
  EXPECT_NOT_OK(error);
  EXPECT_TRUE(result.forces.empty());
}

TEST(TactileObservationTest, ScatterRejectsAccumulationOverflow) {
  DynamicArray<Real3> const positions = {{0_r, 0_r, 0_r}};
  Error error;
  TactileTaxelLayout const layout = MakeTactileTaxelLayout(positions, error);
  EXPECT_OK(error);
  real const large = std::numeric_limits<real>::max();
  DynamicArray<NodeContactForce> const sparse = {
      {.index = 0, .force = {large, 0_r, 0_r}},
      {.index = 0, .force = {large, 0_r, 0_r}},
  };

  TactileTaxelForces const result = ScatterTactileTaxelForces(sparse, TransformRT{}, layout, error);

  EXPECT_NOT_OK(error);
  EXPECT_TRUE(result.forces.empty());
}

TEST(TactileObservationTest, ScatterConvertsWorldForceToSensorFrame) {
  DynamicArray<Real3> const positions = {{0_r, 0_r, 0_r}};
  Error error;
  TactileTaxelLayout const layout = MakeTactileTaxelLayout(positions, error);
  EXPECT_OK(error);
  DynamicArray<NodeContactForce> const sparse = {{.index = 0, .force = {0_r, 1_r, 0_r}}};
  TransformRT const worldFromSensor(Quaternion::RotationZ(1.5707963267948966_r));

  TactileTaxelForces const result =
      ScatterTactileTaxelForces(sparse, worldFromSensor, layout, error);

  EXPECT_OK(error);
  ASSERT_EQ(result.forces.size(), 1);
  EXPECT_NEAR(result.forces[0][0], 1_r, 1e-5_r);
  EXPECT_NEAR(result.forces[0][1], 0_r, 1e-5_r);
  EXPECT_NEAR(result.forces[0][2], 0_r, 1e-5_r);
}

TEST(TactileObservationTest, AggregateIsForceOnlyAndConservesLayerOneSum) {
  TactileTaxelForces layerOne;
  layerOne.forces = {{1_r, 2_r, 3_r}, {-4_r, 5_r, -6_r}, {0.5_r, -0.5_r, 1_r}};
  Error error;

  TactileAggregateForce const layerTwo = AggregateTactileForce(layerOne, error);
  EXPECT_OK(error);
  EXPECT_EQ(layerTwo.force, (Real3{-2.5_r, 6.5_r, -2_r}));
}

TEST(TactileObservationTest, AggregateWrenchPreservesForceAndContactMoment) {
  DynamicArray<Real3> const positions = {{0.1_r, 0_r, 0_r}, {-0.1_r, 0_r, 0_r}};
  Error error;
  TactileTaxelLayout const layout = MakeTactileTaxelLayout(positions, error);
  EXPECT_OK(error);
  TactileTaxelForces layerOne;
  layerOne.forces = {{0_r, 1_r, 0_r}, {0_r, -1_r, 0_r}};

  TactileAggregateWrench const layerTwo = AggregateTactileWrench(layout, layerOne, Real3{}, error);

  EXPECT_OK(error);
  EXPECT_EQ(layerTwo.referencePoint, Real3{});
  EXPECT_NEAR_EQ(layerTwo.force, Real3{});
  EXPECT_NEAR_EQ(layerTwo.torque, (Real3{0_r, 0_r, 0.2_r}));
}

TEST(TactileObservationTest, AggregateWrenchMovesTorqueWithReferencePoint) {
  DynamicArray<Real3> const positions = {{0.1_r, 0_r, 0_r}};
  Error error;
  TactileTaxelLayout const layout = MakeTactileTaxelLayout(positions, error);
  EXPECT_OK(error);
  TactileTaxelForces layerOne;
  layerOne.forces = {{0_r, -2_r, 0_r}};

  TactileAggregateWrench const atOrigin = AggregateTactileWrench(layout, layerOne, Real3{}, error);
  EXPECT_OK(error);
  Real3 const shiftedReference{0_r, 0_r, 0.05_r};
  TactileAggregateWrench const shifted =
      AggregateTactileWrench(layout, layerOne, shiftedReference, error);

  EXPECT_OK(error);
  EXPECT_NEAR_EQ(atOrigin.force, shifted.force);
  EXPECT_NEAR_EQ(atOrigin.torque - Cross(shiftedReference, atOrigin.force), shifted.torque);
}

TEST(TactileObservationTest, HemisphereMapReusesAPreallocatedGrid) {
  DynamicArray<Real3> const positions = {{0_r, 0_r, 1_r}, {1_r, 0_r, 0_r}};
  Error error;
  TactileTaxelLayout const layout = MakeTactileTaxelLayout(positions, error);
  EXPECT_OK(error);
  TactileTaxelForces forces;
  forces.forces = {{1_r, 2_r, 3_r}, {4_r, 5_r, 6_r}};
  TactileHemisphereRectParams const params{.width = 4, .height = 2};
  TactileForceGrid grid;
  grid.forces.resize(8);
  grid.taxelCounts.resize(8);
  Real3* const forceAllocation = grid.forces.data();
  int* const countAllocation = grid.taxelCounts.data();

  MapTactileHemisphereToRectInto(layout, forces, params, grid, error);

  EXPECT_OK(error);
  EXPECT_EQ(grid.forces.data(), forceAllocation);
  EXPECT_EQ(grid.taxelCounts.data(), countAllocation);
  Real3 sum = {};
  for (Real3 const& force : grid.forces) {
    sum += force;
  }
  EXPECT_EQ(sum, (Real3{5_r, 7_r, 9_r}));
}

TEST(TactileObservationTest, RejectsNonFiniteInputs) {
  DynamicArray<Real3> const positions = {{0_r, 0_r, std::numeric_limits<real>::infinity()}};
  Error error;
  TactileTaxelLayout const layout = MakeTactileTaxelLayout(positions, error);
  EXPECT_NOT_OK(error);
  EXPECT_TRUE(layout.positions.empty());
}

TEST(TactileObservationTest, AggregateWrenchRejectsMutatedNonFiniteLayoutPositions) {
  DynamicArray<Real3> const positions = {{0_r, 0_r, 0_r}};
  Error error;
  TactileTaxelLayout layout = MakeTactileTaxelLayout(positions, error);
  EXPECT_OK(error);
  layout.positions[0][0] = std::numeric_limits<real>::infinity();
  TactileTaxelForces layerOne;
  layerOne.forces = {{1_r, 0_r, 0_r}};

  AggregateTactileWrench(layout, layerOne, Real3{}, error);

  EXPECT_NOT_OK(error);
}

TEST(TactileObservationTest, HemisphereRectMapsTipAndRimAndConservesForce) {
  DynamicArray<Real3> const positions = {{0_r, 0_r, 1_r}, {1_r, 0_r, 0_r}, {0_r, 1_r, 0_r}};
  Error error;
  TactileTaxelLayout const layout = MakeTactileTaxelLayout(positions, error);
  EXPECT_OK(error);
  TactileTaxelForces layerOne;
  layerOne.forces = {{1_r, 0_r, 0_r}, {0_r, 2_r, 0_r}, {0_r, 0_r, 3_r}};
  TactileHemisphereRectParams params;
  params.width = 4;
  params.height = 2;
  params.rasterization = TactileRectRasterizationType::NearestCell;

  TactileForceGrid const grid = MapTactileHemisphereToRect(layout, layerOne, params, error);

  EXPECT_OK(error);
  ASSERT_EQ(grid.forces.size(), 8);
  ASSERT_EQ(grid.taxelCounts.size(), 8);
  EXPECT_EQ(grid.forces[2], (Real3{0_r, 0_r, 3_r}));
  EXPECT_EQ(grid.forces[6], (Real3{1_r, 0_r, 0_r}));
  EXPECT_EQ(grid.forces[7], (Real3{0_r, 2_r, 0_r}));
  EXPECT_EQ(grid.taxelCounts[2], 1);
  EXPECT_EQ(grid.taxelCounts[6], 1);
  EXPECT_EQ(grid.taxelCounts[7], 1);
  TactileAggregateForce const layerTwo = AggregateTactileForce(layerOne, error);
  EXPECT_OK(error);
  Real3 gridSum = {};
  for (Real3 const& force : grid.forces) {
    gridSum += force;
  }
  EXPECT_EQ(gridSum, layerTwo.force);
}

TEST(TactileObservationTest, ConservativeVoronoiFillsGridAndConservesForce) {
  DynamicArray<Real3> const positions = {{0_r, 0_r, 1_r}};
  Error error;
  TactileTaxelLayout const layout = MakeTactileTaxelLayout(positions, error);
  EXPECT_OK(error);
  TactileTaxelForces layerOne;
  layerOne.forces = {{16_r, 32_r, 48_r}};
  TactileHemisphereRectParams params;
  params.width = 4;
  params.height = 4;

  TactileRectProjectionMap const projection = BuildTactileRectProjection(layout, params, error);
  EXPECT_OK(error);
  EXPECT_EQ(projection.rasterization, TactileRectRasterizationType::ConservativeVoronoi);
  EXPECT_EQ(projection.voronoiOversampling, 4);
  ASSERT_EQ(projection.taxelContributionOffsets.size(), 2);
  EXPECT_EQ(projection.taxelContributionOffsets[0], 0);
  EXPECT_EQ(projection.taxelContributionOffsets[1], 16);

  TactileForceGrid const grid = RasterizeTactileRectProjection(projection, layerOne, error);
  EXPECT_OK(error);
  ASSERT_EQ(grid.forces.size(), 16);
  Real3 gridSum = {};
  for (int cell = 0; cell < isize(grid.forces); ++cell) {
    EXPECT_NEAR_EQ(grid.forces[cell], (Real3{1_r, 2_r, 3_r}));
    EXPECT_EQ(grid.taxelCounts[cell], 1);
    gridSum += grid.forces[cell];
  }
  EXPECT_NEAR_EQ(gridSum, layerOne.forces[0]);
}

TEST(TactileObservationTest, RectProjectionTypesPlaceTipAtEvenGridCenter) {
  DynamicArray<Real3> const positions = {{0_r, 0_r, 1_r}, {1_r, 0_r, 0_r}, {0_r, 1_r, 0_r}};
  Error error;
  TactileTaxelLayout const layout = MakeTactileTaxelLayout(positions, error);
  EXPECT_OK(error);
  TactileHemisphereRectParams params;
  params.width = 256;
  params.height = 256;
  params.rasterization = TactileRectRasterizationType::NearestCell;

  for (TactileRectProjectionType const projectionType :
       {TactileRectProjectionType::AzimuthalEqualArea,
        TactileRectProjectionType::AzimuthalEquidistant,
        TactileRectProjectionType::Octahedral}) {
    params.projection = projectionType;
    TactileRectProjectionMap const projection = BuildTactileRectProjection(layout, params, error);
    EXPECT_OK(error);
    ASSERT_EQ(projection.coordinates.size(), 3);
    ASSERT_EQ(projection.taxelCells.size(), 3);
    EXPECT_NEAR_EQ(projection.coordinates[0], Real2{});
    EXPECT_EQ(projection.taxelCells[0], 128 * 256 + 128);
    EXPECT_EQ(projection.taxelCells[1], 128 * 256 + 255);
    EXPECT_EQ(projection.taxelCells[2], 128);
    EXPECT_EQ(projection.selectedTaxelCount, 3);
  }
}

TEST(TactileObservationTest, CachedProjectionRasterizesFramesIntoStableStorage) {
  DynamicArray<Real3> const positions = {{0_r, 0_r, 1_r}, {1_r, 0_r, 0_r}};
  Error error;
  TactileTaxelLayout const layout = MakeTactileTaxelLayout(positions, error);
  EXPECT_OK(error);
  TactileHemisphereRectParams params;
  params.width = 4;
  params.height = 2;
  params.rasterization = TactileRectRasterizationType::NearestCell;
  TactileRectProjectionMap const projection = BuildTactileRectProjection(layout, params, error);
  EXPECT_OK(error);

  TactileForceGrid grid;
  grid.forces.resize(8);
  grid.taxelCounts.resize(8);
  Real3 const* const forceStorage = grid.forces.data();
  int const* const countStorage = grid.taxelCounts.data();
  TactileTaxelForces frame;
  frame.forces = {{1_r, 2_r, 3_r}, {4_r, 5_r, 6_r}};

  RasterizeTactileRectProjectionInto(projection, frame, grid, error);
  EXPECT_OK(error);
  frame.forces = {{7_r, 8_r, 9_r}, {10_r, 11_r, 12_r}};
  RasterizeTactileRectProjectionInto(projection, frame, grid, error);

  EXPECT_OK(error);
  EXPECT_EQ(grid.forces.data(), forceStorage);
  EXPECT_EQ(grid.taxelCounts.data(), countStorage);
  Real3 forceSum = {};
  for (Real3 const& force : grid.forces) {
    forceSum += force;
  }
  EXPECT_EQ(forceSum, (Real3{17_r, 19_r, 21_r}));
}

TEST(TactileObservationTest, RectProjectionTypesResolveDistinctInteriorCoordinates) {
  constexpr real kSqrtThreeOverTwo = 0.8660254037844386_r;
  DynamicArray<Real3> const positions = {{kSqrtThreeOverTwo, 0_r, 0.5_r}};
  Error error;
  TactileTaxelLayout const layout = MakeTactileTaxelLayout(positions, error);
  EXPECT_OK(error);
  TactileHemisphereRectParams params;
  params.width = 256;
  params.height = 256;
  params.rasterization = TactileRectRasterizationType::NearestCell;

  params.projection = TactileRectProjectionType::AzimuthalEqualArea;
  TactileRectProjectionMap const equalArea = BuildTactileRectProjection(layout, params, error);
  EXPECT_OK(error);
  params.projection = TactileRectProjectionType::AzimuthalEquidistant;
  TactileRectProjectionMap const equidistant = BuildTactileRectProjection(layout, params, error);
  EXPECT_OK(error);
  params.projection = TactileRectProjectionType::Octahedral;
  TactileRectProjectionMap const octahedral = BuildTactileRectProjection(layout, params, error);
  EXPECT_OK(error);

  EXPECT_GT(equalArea.coordinates[0][0], equidistant.coordinates[0][0]);
  EXPECT_GT(equidistant.coordinates[0][0], octahedral.coordinates[0][0]);
}

TEST(TactileObservationTest, RectProjectionExplicitlyExcludesTaxelsOutsideCap) {
  DynamicArray<Real3> const positions = {{0_r, 0_r, 1_r}, {0_r, 0_r, -1_r}};
  Error error;
  TactileTaxelLayout const layout = MakeTactileTaxelLayout(positions, error);
  EXPECT_OK(error);
  TactileTaxelForces layerOne;
  layerOne.forces = {{1_r, 2_r, 3_r}, {100_r, 200_r, 300_r}};
  TactileHemisphereRectParams params;
  params.width = 4;
  params.height = 4;
  params.rasterization = TactileRectRasterizationType::NearestCell;
  params.outsidePolicy = TactileRectOutsidePolicy::Exclude;

  TactileRectProjectionMap const projection = BuildTactileRectProjection(layout, params, error);
  EXPECT_OK(error);
  ASSERT_EQ(projection.taxelCells.size(), 2);
  EXPECT_EQ(projection.selectedTaxelCount, 1);
  EXPECT_GE(projection.taxelCells[0], 0);
  EXPECT_EQ(projection.taxelCells[1], -1);
  TactileForceGrid const grid = RasterizeTactileRectProjection(projection, layerOne, error);
  EXPECT_OK(error);
  Real3 gridSum = {};
  int countSum = 0;
  for (int cell = 0; cell < isize(grid.forces); ++cell) {
    gridSum += grid.forces[cell];
    countSum += grid.taxelCounts[cell];
  }
  EXPECT_EQ(gridSum, (Real3{1_r, 2_r, 3_r}));
  EXPECT_EQ(countSum, 1);
}

TEST(TactileObservationTest, HemisphereRectProjectionParametersChangeTheMapping) {
  DynamicArray<Real3> const positions = {{0_r, 0_r, 1_r}, {1_r, 0_r, 0_r}};
  Error error;
  TactileTaxelLayout const layout = MakeTactileTaxelLayout(positions, error);
  EXPECT_OK(error);
  TactileTaxelForces layerOne;
  layerOne.forces = {{1_r, 0_r, 0_r}, {0_r, 1_r, 0_r}};
  TactileHemisphereRectParams params;
  params.width = 4;
  params.height = 2;

  TactileForceGrid const first = MapTactileHemisphereToRect(layout, layerOne, params, error);
  EXPECT_OK(error);
  TactileForceGrid const second = MapTactileHemisphereToRect(layout, layerOne, params, error);
  EXPECT_OK(error);
  params.rasterization = TactileRectRasterizationType::NearestCell;
  TactileForceGrid const nearest = MapTactileHemisphereToRect(layout, layerOne, params, error);
  EXPECT_OK(error);
  params.rasterization = TactileRectRasterizationType::ConservativeVoronoi;
  params.voronoiOversampling = 5;
  TactileForceGrid const oversampled = MapTactileHemisphereToRect(layout, layerOne, params, error);
  EXPECT_OK(error);
  params.voronoiOversampling = 4;
  params.width = 5;
  TactileForceGrid const changed = MapTactileHemisphereToRect(layout, layerOne, params, error);

  EXPECT_OK(error);
  EXPECT_EQ(first.forces, second.forces);
  EXPECT_NE(first.taxelCounts, nearest.taxelCounts);
  EXPECT_EQ(first.forces.size(), oversampled.forces.size());
  EXPECT_NE(first.forces.size(), changed.forces.size());
}

TEST(TactileObservationTest, HemisphereRectWrapsPositivePiLongitudeToTheFirstColumn) {
  DynamicArray<Real3> const positions = {{-1_r, 0_r, 0_r}};
  Error error;
  TactileTaxelLayout const layout = MakeTactileTaxelLayout(positions, error);
  EXPECT_OK(error);
  TactileTaxelForces layerOne;
  layerOne.forces = {{1_r, 2_r, 3_r}};
  TactileHemisphereRectParams params;
  params.width = 4;
  params.height = 2;
  params.rasterization = TactileRectRasterizationType::NearestCell;

  TactileForceGrid const grid = MapTactileHemisphereToRect(layout, layerOne, params, error);

  EXPECT_OK(error);
  // The projected vertical coordinate is zero, so rounding may select either
  // adjacent row. The longitude seam must still wrap to the first column.
  EXPECT_EQ(grid.taxelCounts[0] + grid.taxelCounts[4], 1);
  int const populatedCell = grid.taxelCounts[0] == 1 ? 0 : 4;
  EXPECT_EQ(grid.forces[populatedCell], (Real3{1_r, 2_r, 3_r}));
}

TEST(TactileObservationTest, HemisphereRectRejectsTaxelBehindRim) {
  DynamicArray<Real3> const positions = {{0_r, 0_r, -1_r}};
  Error error;
  TactileTaxelLayout const layout = MakeTactileTaxelLayout(positions, error);
  EXPECT_OK(error);
  TactileTaxelForces layerOne;
  layerOne.forces = {{0_r, 0_r, 0_r}};
  TactileHemisphereRectParams params;
  params.width = 4;
  params.height = 2;

  TactileForceGrid const grid = MapTactileHemisphereToRect(layout, layerOne, params, error);

  EXPECT_NOT_OK(error);
  EXPECT_TRUE(grid.forces.empty());
}

TEST(TactileObservationTest, HemisphereRectRejectsOverflowingGridShape) {
  DynamicArray<Real3> const positions = {{0_r, 0_r, 1_r}};
  Error error;
  TactileTaxelLayout const layout = MakeTactileTaxelLayout(positions, error);
  EXPECT_OK(error);
  TactileTaxelForces layerOne;
  layerOne.forces = {{1_r, 0_r, 0_r}};
  TactileHemisphereRectParams params;
  params.width = std::numeric_limits<int>::max();
  params.height = 2;

  TactileForceGrid const grid = MapTactileHemisphereToRect(layout, layerOne, params, error);

  EXPECT_NOT_OK(error);
  EXPECT_TRUE(grid.forces.empty());
}

TEST(TactileObservationTest, RectProjectionSupportsSingleCellDimensions) {
  DynamicArray<Real3> const positions = {{0_r, 0_r, 1_r}};
  Error layoutError;
  TactileTaxelLayout const layout = MakeTactileTaxelLayout(positions, layoutError);
  EXPECT_OK(layoutError);
  TactileHemisphereRectParams params;
  params.width = 1;
  params.height = 4;

  Error narrowError;
  TactileRectProjectionMap const narrow = BuildTactileRectProjection(layout, params, narrowError);
  EXPECT_OK(narrowError);
  ASSERT_EQ(narrow.taxelCells.size(), 1);
  EXPECT_EQ(narrow.taxelCells[0], 2);
  EXPECT_EQ(narrow.contributionCells.size(), 4);

  params.width = 4;
  params.height = 1;
  Error shortError;
  TactileRectProjectionMap const shortMap = BuildTactileRectProjection(layout, params, shortError);
  EXPECT_OK(shortError);
  ASSERT_EQ(shortMap.taxelCells.size(), 1);
  EXPECT_EQ(shortMap.taxelCells[0], 2);
  EXPECT_EQ(shortMap.contributionCells.size(), 4);

  params.width = 1;
  TactileTaxelForces layerOne;
  layerOne.forces = {{1_r, 2_r, 3_r}};
  Error singleError;
  TactileForceGrid const single = MapTactileHemisphereToRect(layout, layerOne, params, singleError);
  EXPECT_OK(singleError);
  ASSERT_EQ(single.forces.size(), 1);
  EXPECT_EQ(single.forces[0], (Real3{1_r, 2_r, 3_r}));
}

TEST(TactileObservationTest, HemisphereRectRejectsMutatedNonFiniteLayoutPositions) {
  DynamicArray<Real3> const positions = {{0_r, 0_r, 1_r}};
  Error error;
  TactileTaxelLayout layout = MakeTactileTaxelLayout(positions, error);
  EXPECT_OK(error);
  layout.positions[0][0] = std::numeric_limits<real>::infinity();
  TactileTaxelForces layerOne;
  layerOne.forces = {{1_r, 0_r, 0_r}};
  TactileHemisphereRectParams params;
  params.width = 4;
  params.height = 2;

  TactileForceGrid const grid = MapTactileHemisphereToRect(layout, layerOne, params, error);

  EXPECT_NOT_OK(error);
  EXPECT_TRUE(grid.forces.empty());
}

TEST(TactileObservationTest, ReaderRejectsAnIntegrationStageBehindTheOutputState) {
  std::unique_ptr<Context, decltype(&mochi::DestroyContext)> context{
      mochi::CreateContext(0), mochi::DestroyContext};
  ASSERT_NE(context, nullptr);
  Scene* const scene = context->CreateScene("tactile_query_integration");
  ASSERT_NE(scene, nullptr);
  auto&& [coordinates, connectivity] = mochi::test::CreateMinimalTetMeshUnitCube();
  auto shape = context->CreateTetMeshShape(
      Flatten(MakeSpan(coordinates)), Flatten(MakeSpan(connectivity)), mochi::test::ExpectOK{});
  RigidActorParams rigidParams;
  rigidParams.shape = shape;
  rigidParams.density = 1000_r;
  Actor* const actor = scene->CreateRigidActor(rigidParams, mochi::test::ExpectOK{});
  ASSERT_NE(actor, nullptr);
  auto solver = scene->GetSolverParams();
  solver.integrationMethod = IntegrationMethod::DIRK23;
  scene->SetSolverParams(solver, mochi::test::ExpectOK{});
  auto const layout = MakeTactileTaxelLayout(MakeSpan(coordinates), mochi::test::ExpectOK{});
  TactileTaxelForces out;
  out.forces.resize(layout.positions.size(), Real3{1_r, 2_r, 3_r});
  auto const* const allocation = out.forces.data();
  Error error;

  ReadTactileTaxelForcesInto(actor, TransformRT{}, layout, out, error);

  EXPECT_NOT_OK(error);
  EXPECT_NE(std::string{error.ToString()}.find("last integration stage"), std::string::npos);
  EXPECT_EQ(allocation, out.forces.data());
  for (auto const& force : out.forces) {
    EXPECT_EQ(force, (Real3{1_r, 2_r, 3_r}));
  }
}
