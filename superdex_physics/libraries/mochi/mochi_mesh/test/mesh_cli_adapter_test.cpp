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

#include "mesh_cli_adapter.h"

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/span_utils.h>

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <limits>
#include <vector>

using namespace mochi;
using namespace mochi::mesh;

TEST(MeshCliAdapterTest, MeshRoundTripTriMesh) {
  MeshData mesh;
  mesh.nodesPerElement = 3;
  mesh.coordinates = {0_r, 0_r, 0_r, 1_r, 0_r, 0_r, 0_r, 1_r, 0_r};
  mesh.connectivity = {0, 1, 2};
  auto cliMesh = cli_adapter::ToCliMeshData(mesh, test::ExpectOK{});
  EXPECT_EQ(3, cliMesh.nodesPerElement);
  EXPECT_EQ(cliMesh.coordinates, (std::vector<double>{0, 0, 0, 1, 0, 0, 0, 1, 0}));
  EXPECT_EQ(cliMesh.connectivity, (std::vector<int32_t>{0, 1, 2}));
  MeshData const result = cli_adapter::FromCliMeshData(cliMesh, test::ExpectOK{});
  EXPECT_EQ(result.coordinates, mesh.coordinates);
  EXPECT_EQ(result.connectivity, mesh.connectivity);
}

TEST(MeshCliAdapterTest, MeshRoundTripTetMesh) {
  MeshData mesh;
  mesh.nodesPerElement = 4;
  mesh.coordinates = {0_r, 0_r, 0_r, 1_r, 0_r, 0_r, 0_r, 1_r, 0_r, 0_r, 0_r, 1_r};
  mesh.connectivity = {0, 1, 2, 3};
  auto cliMesh = cli_adapter::ToCliMeshData(mesh, test::ExpectOK{});
  EXPECT_EQ(4, cliMesh.nodesPerElement);
  EXPECT_EQ(cliMesh.coordinates, (std::vector<double>{0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1}));
  EXPECT_EQ(cliMesh.connectivity, (std::vector<int32_t>{0, 1, 2, 3}));
  MeshData const result = cli_adapter::FromCliMeshData(cliMesh, test::ExpectOK{});
  EXPECT_EQ(result.coordinates, mesh.coordinates);
  EXPECT_EQ(result.connectivity, mesh.connectivity);
}

TEST(MeshCliAdapterTest, MeshRejectsInvalidConnectivity) {
  MeshData mesh;
  mesh.nodesPerElement = 3;
  mesh.coordinates = {0_r, 0_r, 0_r};
  mesh.connectivity = {0, 1, 0};
  [[maybe_unused]] auto cliMesh = cli_adapter::ToCliMeshData(mesh, test::ExpectNotOK{});
}

TEST(MeshCliAdapterTest, ScalarFieldMapsDimsValuesAndBounds) {
  GridSdfData grid;
  grid.dims = {2, 2, 2};
  grid.values.resize(8, 1_r);
  grid.bounds = Aabb(Real3{0_r, 0_r, 0_r}, Real3{1_r, 2_r, 3_r});
  grid.negativeValueBounds = Aabb(Real3{0.25_r, 0.5_r, 0.75_r}, Real3{0.75_r, 1.5_r, 2.25_r});

  cli::ScalarField3d const field = cli_adapter::ToCliScalarField(grid, test::ExpectOK{});
  EXPECT_EQ(field.dims, (std::array<int32_t, 3>{2, 2, 2}));
  EXPECT_EQ(field.values, std::vector<double>(8, 1.0));
  EXPECT_EQ(field.boundsMin, (std::array<double, 3>{0.0, 0.0, 0.0}));
  EXPECT_EQ(field.boundsMax, (std::array<double, 3>{1.0, 2.0, 3.0}));
  EXPECT_EQ(field.negativeValueBoundsMin, (std::array<double, 3>{0.25, 0.5, 0.75}));
  EXPECT_EQ(field.negativeValueBoundsMax, (std::array<double, 3>{0.75, 1.5, 2.25}));
}

TEST(MeshCliAdapterTest, ScalarFieldRoundTripsGridSdfData) {
  GridSdfData grid;
  grid.dims = {2, 2, 2};
  grid.values = {0_r, 1_r, 2_r, 3_r, 4_r, 5_r, 6_r, 7_r};
  grid.bounds = Aabb(Real3{-1_r, 0_r, 0.5_r}, Real3{1_r, 2_r, 3_r});
  // Deliberately tighter than bounds on every axis, so a dropped negativeValueBounds would decode
  // as the outer box (or as the degenerate default) and fail the comparison below.
  grid.negativeValueBounds = Aabb(Real3{-0.5_r, 0.25_r, 0.75_r}, Real3{0.5_r, 1.5_r, 2.5_r});

  cli::ScalarField3d const field = cli_adapter::ToCliScalarField(grid, test::ExpectOK{});
  GridSdfData const result = cli_adapter::FromCliScalarField(field, test::ExpectOK{});
  for (int axis = 0; axis < 3; ++axis) {
    EXPECT_EQ(result.dims[axis], grid.dims[axis]);
    EXPECT_EQ(result.bounds.GetMin()[axis], grid.bounds.GetMin()[axis]);
    EXPECT_EQ(result.bounds.GetMax()[axis], grid.bounds.GetMax()[axis]);
    EXPECT_EQ(result.negativeValueBounds.GetMin()[axis], grid.negativeValueBounds.GetMin()[axis]);
    EXPECT_EQ(result.negativeValueBounds.GetMax()[axis], grid.negativeValueBounds.GetMax()[axis]);
  }
  EXPECT_SPAN_EQ(MakeConstSpan(result.values), MakeConstSpan(grid.values));
}

// The sampled volume is often padded outward past the surface, so the negative-value box is a
// distinct value that has to survive the trip rather than being inferred from the outer bounds.
// It is carried verbatim: callers that do not populate it (isosurface reconstruction, for one)
// must still be able to send a field.
TEST(MeshCliAdapterTest, ScalarFieldAcceptsUnpopulatedNegativeValueBounds) {
  GridSdfData grid;
  grid.dims = {2, 2, 2};
  grid.values.resize(8, 1_r);
  grid.bounds = Aabb(Real3{0_r, 0_r, 0_r}, Real3{1_r, 1_r, 1_r});

  cli::ScalarField3d const field = cli_adapter::ToCliScalarField(grid, test::ExpectOK{});
  EXPECT_EQ(field.negativeValueBoundsMin, (std::array<double, 3>{0.0, 0.0, 0.0}));
  EXPECT_EQ(field.negativeValueBoundsMax, (std::array<double, 3>{0.0, 0.0, 0.0}));
}

TEST(MeshCliAdapterTest, ScalarFieldResponseRejectsMismatchedValueCount) {
  cli::ScalarField3d field;
  field.dims = {2, 2, 2};
  field.values.assign(4, 0.0); // dims call for 8
  field.boundsMin = {0.0, 0.0, 0.0};
  field.boundsMax = {1.0, 1.0, 1.0};
  EXPECT_TRUE(cli_adapter::FromCliScalarField(field, test::ExpectNotOK{}).values.empty());
}

TEST(MeshCliAdapterTest, ScalarFieldResponseRejectsNonFiniteNegativeValueBounds) {
  cli::ScalarField3d field;
  field.dims = {2, 2, 2};
  field.values.assign(8, 0.0);
  field.boundsMin = {0.0, 0.0, 0.0};
  field.boundsMax = {1.0, 1.0, 1.0};
  field.negativeValueBoundsMax = {std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0};
  EXPECT_TRUE(cli_adapter::FromCliScalarField(field, test::ExpectNotOK{}).values.empty());
}

TEST(MeshCliAdapterTest, ScalarFieldRejectsOverflowingDimensions) {
  GridSdfData grid;
  grid.dims = {2097152, 2097152, 4194304};
  grid.bounds = Aabb(Real3{0_r, 0_r, 0_r}, Real3{1_r, 1_r, 1_r});
  [[maybe_unused]] auto field = cli_adapter::ToCliScalarField(grid, test::ExpectNotOK{});
}

#if !MOCHI_USE_DOUBLE_PRECISION
TEST(MeshCliAdapterTest, ResponseRejectsCoordinateOverflow) {
  cli::MeshData protocol{3, {0, 0, 0, std::numeric_limits<double>::max(), 0, 0}, {0, 1, 0}};
  EXPECT_TRUE(cli_adapter::FromCliMeshData(protocol, test::ExpectNotOK{}).coordinates.empty());
}
#endif
