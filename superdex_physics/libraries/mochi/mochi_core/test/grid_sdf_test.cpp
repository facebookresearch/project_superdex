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

#include <mochi_core/geometry/grid_sdf.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/nd_array_utils.h>

#include <limits>
#include <memory>

using namespace mochi;

// Non-uniform box mesh: AABB min=(0,0,0), max=(1,2,3), range=(1,2,3)
// 18 unique edges: 4x1, 4x2, 4x3, 2xsqrt(5), 2xsqrt(10), 2xsqrt(13)
// Topologically closed.
static auto CreateTestMesh() {
  return std::make_shared<TriangularMesh>(test::CreateMinimalTriMeshUnitCube(Real3{1_r, 2_r, 3_r}));
}

// Base params that disable padding and min-resolution clamping so that
// the cell resolution is determined solely by the resolution mode and delta.
static GridSdfParams CreateBaseParams() {
  GridSdfParams params;
  params.boundaryPaddingDist = 0_r;
  params.minGridResolution = Int3{1, 1, 1};
  return params;
}

// Independently compute the expected cell resolution from a mesh AABB and delta,
// using the same well-known formula: Ceil(gridSize / delta), clamped to minGridResolution.
// This mirrors the public specification in grid_sdf_params.h without depending on GridSdf
// internals.
static Int3 ExpectedCellResolution(
    Aabb const& meshAabb,
    Real3 const& delta,
    real boundaryPaddingDist,
    Int3 const& minGridResolution) {
  Aabb gridBounds = ExpandShape(meshAabb, boundaryPaddingDist);
  Real3 epsilon = gridBounds.GetSize() * std::numeric_limits<real>::epsilon();
  gridBounds = Aabb{gridBounds.GetMin() - epsilon, gridBounds.GetMax() + epsilon};
  Int3 cellDims = StaticCast<Int3>(Ceil(gridBounds.GetSize() / delta));
  for (int i = 0; i < 3; ++i) {
    cellDims[i] = Max(cellDims[i], Max(1, minGridResolution[i]));
  }
  return cellDims;
}

// ---------------------------------------------------------------------------
// Axis-Based Resolution Mode Tests
// ---------------------------------------------------------------------------

TEST(GridSdf, Resolution_LargestAxis) {
  auto mesh = CreateTestMesh();
  auto params = CreateBaseParams();
  params.resolutionMode = GridSdfResolutionMode::LargestAxis;
  params.resolutionDelta = Real3{0.5_r, 0.5_r, 0.5_r};

  // Reference = max(1,2,3) = 3, delta = 0.5 * 3 = 1.5
  real const reference = Max(mesh->GetAabb().GetSize());
  Real3 const delta = params.resolutionDelta * reference;
  Int3 const expected = ExpectedCellResolution(
      mesh->GetAabb(), delta, params.boundaryPaddingDist, params.minGridResolution);

  GridSdf sdf(mesh, params, test::ExpectOK{});
  EXPECT_EQ(sdf.GetCellResolution(), expected);
}

TEST(GridSdf, Resolution_SmallestAxis) {
  auto mesh = CreateTestMesh();
  auto params = CreateBaseParams();
  params.resolutionMode = GridSdfResolutionMode::SmallestAxis;
  params.resolutionDelta = Real3{0.5_r, 0.5_r, 0.5_r};

  // Reference = min(1,2,3) = 1, delta = 0.5 * 1 = 0.5
  real const reference = Min(mesh->GetAabb().GetSize());
  Real3 const delta = params.resolutionDelta * reference;
  Int3 const expected = ExpectedCellResolution(
      mesh->GetAabb(), delta, params.boundaryPaddingDist, params.minGridResolution);

  GridSdf sdf(mesh, params, test::ExpectOK{});
  EXPECT_EQ(sdf.GetCellResolution(), expected);
}

TEST(GridSdf, Resolution_MeanAxis) {
  auto mesh = CreateTestMesh();
  auto params = CreateBaseParams();
  params.resolutionMode = GridSdfResolutionMode::MeanAxis;
  params.resolutionDelta = Real3{0.5_r, 0.5_r, 0.5_r};

  // Reference = mean(1,2,3) = 2, delta = 0.5 * 2 = 1.0
  real const reference = Mean(mesh->GetAabb().GetSize());
  Real3 const delta = params.resolutionDelta * reference;
  Int3 const expected = ExpectedCellResolution(
      mesh->GetAabb(), delta, params.boundaryPaddingDist, params.minGridResolution);

  GridSdf sdf(mesh, params, test::ExpectOK{});
  EXPECT_EQ(sdf.GetCellResolution(), expected);
}

// ---------------------------------------------------------------------------
// Edge-Based Resolution Mode Tests
// ---------------------------------------------------------------------------

TEST(GridSdf, Resolution_LargestEdge) {
  auto mesh = CreateTestMesh();
  auto params = CreateBaseParams();
  params.resolutionMode = GridSdfResolutionMode::LargestEdge;
  params.resolutionDelta = Real3{0.5_r, 0.5_r, 0.5_r};

  // Independently find the largest edge length: sqrt(13) ~ 3.606
  auto nodes = mesh->GetNodeCoordinates();
  auto edges = mesh->GetEdges();
  real largestEdge = 0_r;
  for (int i = 0; i < mesh->GetNumEdges(); ++i) {
    largestEdge = Max(largestEdge, Norm(nodes[edges[i][0]] - nodes[edges[i][1]]));
  }

  Real3 const delta = params.resolutionDelta * largestEdge;
  Int3 const expected = ExpectedCellResolution(
      mesh->GetAabb(), delta, params.boundaryPaddingDist, params.minGridResolution);

  GridSdf sdf(mesh, params, test::ExpectOK{});
  EXPECT_EQ(sdf.GetCellResolution(), expected);
}

TEST(GridSdf, Resolution_SmallestEdge) {
  auto mesh = CreateTestMesh();
  auto params = CreateBaseParams();
  params.resolutionMode = GridSdfResolutionMode::SmallestEdge;
  params.resolutionDelta = Real3{0.5_r, 0.5_r, 0.5_r};

  // Independently find the smallest edge length: 1.0
  auto nodes = mesh->GetNodeCoordinates();
  auto edges = mesh->GetEdges();
  real smallestEdge = std::numeric_limits<real>::max();
  for (int i = 0; i < mesh->GetNumEdges(); ++i) {
    smallestEdge = Min(smallestEdge, Norm(nodes[edges[i][0]] - nodes[edges[i][1]]));
  }

  Real3 const delta = params.resolutionDelta * smallestEdge;
  Int3 const expected = ExpectedCellResolution(
      mesh->GetAabb(), delta, params.boundaryPaddingDist, params.minGridResolution);

  GridSdf sdf(mesh, params, test::ExpectOK{});
  EXPECT_EQ(sdf.GetCellResolution(), expected);
}

TEST(GridSdf, Resolution_MeanEdge) {
  auto mesh = CreateTestMesh();

  // Independently compute the mean edge length from mesh data
  auto nodes = mesh->GetNodeCoordinates();
  auto edges = mesh->GetEdges();
  real totalEdgeLength = 0_r;
  for (int i = 0; i < mesh->GetNumEdges(); ++i) {
    totalEdgeLength += Norm(nodes[edges[i][0]] - nodes[edges[i][1]]);
  }
  real const meanEdge = totalEdgeLength / static_cast<real>(mesh->GetNumEdges());

  auto params = CreateBaseParams();
  params.resolutionMode = GridSdfResolutionMode::MeanEdge;
  params.resolutionDelta = Real3{0.5_r, 0.5_r, 0.5_r};

  Real3 const delta = params.resolutionDelta * meanEdge;
  Int3 const expected = ExpectedCellResolution(
      mesh->GetAabb(), delta, params.boundaryPaddingDist, params.minGridResolution);

  GridSdf sdf(mesh, params, test::ExpectOK{});
  EXPECT_EQ(sdf.GetCellResolution(), expected);
}

// ---------------------------------------------------------------------------
// Explicit Resolution Mode Test
// ---------------------------------------------------------------------------

TEST(GridSdf, Resolution_Explicit) {
  auto mesh = CreateTestMesh();
  auto params = CreateBaseParams();
  params.resolutionMode = GridSdfResolutionMode::Explicit;
  params.resolutionDelta = Real3{1.0_r, 1.0_r, 1.0_r};

  // In Explicit mode, delta = resolutionDelta directly
  Real3 const delta = params.resolutionDelta;
  Int3 const expected = ExpectedCellResolution(
      mesh->GetAabb(), delta, params.boundaryPaddingDist, params.minGridResolution);

  GridSdf sdf(mesh, params, test::ExpectOK{});
  EXPECT_EQ(sdf.GetCellResolution(), expected);
}

// ---------------------------------------------------------------------------
// Boundary Padding Test
// ---------------------------------------------------------------------------

TEST(GridSdf, BoundaryPadding) {
  auto mesh = CreateTestMesh();

  // Baseline: zero padding
  auto baseParams = CreateBaseParams();
  baseParams.resolutionMode = GridSdfResolutionMode::MeanAxis;
  baseParams.resolutionDelta = Real3{0.5_r, 0.5_r, 0.5_r};
  GridSdf baseline(mesh, baseParams, test::ExpectOK{});

  // With boundary padding
  auto paddedParams = baseParams;
  paddedParams.boundaryPaddingDist = 0.5_r;
  GridSdf padded(mesh, paddedParams, test::ExpectOK{});

  real const tol = 1e-5_r;

  // Collider bounds (negative-value bounds) should approximate the mesh AABB
  auto const& colliderBounds = padded.GetDistanceGrid().GetNegativeValueBounds();
  EXPECT_NEAR_TOL(colliderBounds.GetMin(), (Real3{0_r, 0_r, 0_r}), tol);
  EXPECT_NEAR_TOL(colliderBounds.GetMax(), (Real3{1_r, 2_r, 3_r}), tol);

  // Grid bounds should be expanded by ~boundaryPaddingDist in each direction
  auto const& gridBounds = padded.GetDistanceGrid().GetBounds();
  EXPECT_NEAR_TOL(gridBounds.GetMin(), (Real3{-0.5_r, -0.5_r, -0.5_r}), tol);
  EXPECT_NEAR_TOL(gridBounds.GetMax(), (Real3{1.5_r, 2.5_r, 3.5_r}), tol);

  // Cell resolution should increase with padding vs zero-padding baseline
  Int3 const baseRes = baseline.GetCellResolution();
  Int3 const paddedRes = padded.GetCellResolution();
  for (int i = 0; i < 3; ++i) {
    EXPECT_GT(paddedRes[i], baseRes[i]);
  }
}
