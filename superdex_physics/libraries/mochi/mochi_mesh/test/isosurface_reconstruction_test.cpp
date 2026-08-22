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

#include <mochi_core/geometry/model_data.h>
#include <mochi_core/geometry/model_utils.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/span.h>
#include <mochi_mesh/isosurface_reconstruction.h>
#include <mochi_mesh/mesh_statistics.h>
#include <mochi_mesh/surface_remeshing.h>

#include <algorithm>
#include <cmath>
#include <limits>

using namespace mochi;
using namespace mochi::mesh;
using namespace mochi::test;

TEST(IsosurfaceReconstruction, SphereSdf) {
  // Construct an analytical sphere SDF: signed distance to a sphere of radius 0.4 centered at
  // (0.5, 0.5, 0.5) on a unit cube grid.
  int constexpr kRes = 32;
  real constexpr kRadius = 0.4_r;
  Real3 constexpr kCenter = {0.5_r, 0.5_r, 0.5_r};

  GridSdfData sdfData;
  sdfData.dims = {kRes, kRes, kRes};
  sdfData.bounds = Aabb(Real3{0_r, 0_r, 0_r}, Real3{1_r, 1_r, 1_r});
  sdfData.values.resize(kRes * kRes * kRes);

  for (int x = 0; x < kRes; ++x) {
    for (int y = 0; y < kRes; ++y) {
      for (int z = 0; z < kRes; ++z) {
        Real3 const pos = {
            StaticCast<real>(x) / StaticCast<real>(kRes - 1),
            StaticCast<real>(y) / StaticCast<real>(kRes - 1),
            StaticCast<real>(z) / StaticCast<real>(kRes - 1)};
        real const dist = Norm(pos - kCenter) - kRadius;
        int const idx = kRes * kRes * x + kRes * y + z;
        sdfData.values[idx] = dist;
      }
    }
  }

  ExpectOK expectOK;
  MeshData const result = ReconstructSurfaceFromSdf(sdfData, expectOK);

  EXPECT_GT(result.GetNumNodes(), 0);
  EXPECT_GT(result.GetNumElements(), 0);
  EXPECT_EQ(result.nodesPerElement, 3);

  // Verify the reconstructed mesh is closed
  MeshStatistics const stats = ComputeMeshStatistics(result, nullptr, expectOK);
  EXPECT_TRUE(stats.isClosed);

  // Verify geometric accuracy: all vertices should be near the sphere surface
  for (int i = 0; i < result.GetNumNodes(); ++i) {
    Real3 const v = {
        result.coordinates[3 * i], result.coordinates[3 * i + 1], result.coordinates[3 * i + 2]};
    real const distFromSurface = Abs(Norm(v - kCenter) - kRadius);
    EXPECT_LT(distFromSurface, 0.05_r);
  }
}

TEST(IsosurfaceReconstruction, RoundTripQuality) {
  // Bake SDF from a cube mesh, reconstruct, and verify the Hausdorff distance is reasonable.
  MeshData const cubeMesh = CreateCubeTriMesh();

  ModelData model;
  model.mesh = cubeMesh;

  GridSdfParams sdfParams;
  sdfParams.resolutionMode = GridSdfResolutionMode::Explicit;
  sdfParams.resolutionDelta = {0.05_r, 0.05_r, 0.05_r};

  ExpectOK expectOK;
  model::BakeSdf(model, sdfParams, expectOK);
  ASSERT_TRUE(model.sdf.has_value());

  MeshData const recon = ReconstructSurfaceFromSdf(*model.sdf, expectOK);

  EXPECT_GT(recon.GetNumNodes(), 0);
  EXPECT_GT(recon.GetNumElements(), 0);

  // Measure Hausdorff distance between reconstructed and original
  MeshDataView const refView(cubeMesh);
  MeshStatistics const stats = ComputeMeshStatistics(recon, &refView, expectOK);
  EXPECT_GE(stats.hausdorffDistance, 0.0);
  EXPECT_LT(stats.hausdorffDistance, 0.15);
}

TEST(IsosurfaceReconstruction, ErrorInvalidDims) {
  GridSdfData sdfData;
  sdfData.dims = {1, 1, 1};
  sdfData.values.resize(1);
  sdfData.bounds = Aabb(Real3{0_r, 0_r, 0_r}, Real3{1_r, 1_r, 1_r});

  ExpectNotOK expectNotOK;
  (void)ReconstructSurfaceFromSdf(sdfData, expectNotOK);
}

TEST(IsosurfaceReconstruction, ErrorMismatchedValueCount) {
  GridSdfData sdfData;
  sdfData.dims = {4, 4, 4};
  sdfData.values.resize(10);
  sdfData.bounds = Aabb(Real3{0_r, 0_r, 0_r}, Real3{1_r, 1_r, 1_r});

  ExpectNotOK expectNotOK;
  (void)ReconstructSurfaceFromSdf(sdfData, expectNotOK);
}

TEST(IsosurfaceReconstruction, ErrorDegenerateBounds) {
  GridSdfData sdfData;
  sdfData.dims = {4, 4, 4};
  sdfData.values.resize(64);
  sdfData.bounds = Aabb(Real3{1_r, 0_r, 0_r}, Real3{0_r, 1_r, 1_r});

  ExpectNotOK expectNotOK;
  (void)ReconstructSurfaceFromSdf(sdfData, expectNotOK);
}

TEST(IsosurfaceReconstruction, ErrorNoZeroCrossing) {
  GridSdfData sdfData;
  sdfData.dims = {4, 4, 4};
  sdfData.values.resize(64);
  for (int i = 0; i < 64; ++i) {
    sdfData.values[i] = 1.0_r;
  }
  sdfData.bounds = Aabb(Real3{0_r, 0_r, 0_r}, Real3{1_r, 1_r, 1_r});

  ExpectNotOK expectNotOK;
  (void)ReconstructSurfaceFromSdf(sdfData, expectNotOK);
}

TEST(IsosurfaceReconstruction, SdfRoundTripQuality_AlphaWrap) {
  MeshData const cubeMesh = CreateCubeTriMesh();

  // Remesh the cube with improved AlphaWrap
  SurfaceRemeshingParams remeshParams;
  remeshParams.method = RemeshMethod::AlphaWrap;
  remeshParams.edgeSize = 0.5;
  remeshParams.relativeToMeshSize = false;
  remeshParams.smoothingIterations = 3;
  remeshParams.tangentialRelaxationIterations = 3;
  remeshParams.relaxationStepsPerIteration = 3;

  ExpectOK expectOK;
  MeshData remeshed = RemeshSurface(cubeMesh, remeshParams, expectOK);
  EXPECT_GT(remeshed.GetNumNodes(), 0);

  // Bake SDF from the remeshed mesh
  ModelData model;
  model.mesh = remeshed;
  GridSdfParams sdfParams;
  model::BakeSdf(model, sdfParams, expectOK);
  ASSERT_TRUE(model.sdf.has_value());

  // Reconstruct isosurface from SDF
  GridSdfDataView const sdfView(*model.sdf);
  MeshData const reconstructed = ReconstructSurfaceFromSdf(sdfView, expectOK);
  EXPECT_GT(reconstructed.GetNumNodes(), 0);
  EXPECT_GT(reconstructed.GetNumElements(), 0);

  // Verify the reconstructed mesh is closed (watertight)
  MeshStatistics const reconStats = ComputeMeshStatistics(reconstructed, nullptr, expectOK);
  EXPECT_TRUE(reconStats.isClosed);

  // Verify the Hausdorff distance to the original remeshed mesh is small
  MeshDataView const remeshedView(remeshed);
  MeshStatistics const distStats = ComputeMeshStatistics(reconstructed, &remeshedView, expectOK);
  EXPECT_LT(distStats.hausdorffDistance, 0.15);
}

TEST(IsosurfaceReconstruction, SdfRoundTripQuality_SurfaceDelaunay) {
  MeshData const cubeMesh = CreateCubeTriMesh();

  SurfaceRemeshingParams remeshParams;
  remeshParams.method = RemeshMethod::SurfaceDelaunay;
  remeshParams.edgeSize = 0.5;
  remeshParams.relativeToMeshSize = false;
  remeshParams.facetAngleBound = 20.0;

  ExpectOK expectOK;
  MeshData remeshed = RemeshSurface(cubeMesh, remeshParams, expectOK);
  EXPECT_GT(remeshed.GetNumNodes(), 0);

  ModelData model;
  model.mesh = remeshed;
  GridSdfParams sdfParams;
  model::BakeSdf(model, sdfParams, expectOK);
  ASSERT_TRUE(model.sdf.has_value());

  GridSdfDataView const sdfView(*model.sdf);
  MeshData const reconstructed = ReconstructSurfaceFromSdf(sdfView, expectOK);
  EXPECT_GT(reconstructed.GetNumNodes(), 0);

  MeshStatistics const reconStats = ComputeMeshStatistics(reconstructed, nullptr, expectOK);
  EXPECT_TRUE(reconStats.isClosed);

  MeshDataView const remeshedView(remeshed);
  MeshStatistics const distStats = ComputeMeshStatistics(reconstructed, &remeshedView, expectOK);
  EXPECT_LT(distStats.hausdorffDistance, 0.15);
}

TEST(IsosurfaceReconstruction, AnisotropicGridOffCenterBox) {
  // Anisotropic dims + off-center bounds + axis-asymmetric box.
  // Catches axis permutation, sample/cell off-by-one, and world-coord placement bugs.
  Int3 const dims = {16, 24, 32};
  Real3 const boundsMin = {-2.0_r, 1.0_r, 5.0_r};
  Real3 const boundsMax = {-1.0_r, 3.0_r, 7.0_r};

  // Box centered at midpoint of bounds with half-extents (0.1, 0.2, 0.3).
  Real3 const center = {-1.5_r, 2.0_r, 6.0_r};
  Real3 const halfExtents = {0.1_r, 0.2_r, 0.3_r};

  GridSdfData sdfData;
  sdfData.dims = dims;
  sdfData.bounds = Aabb(boundsMin, boundsMax);
  sdfData.values.resize(dims[0] * dims[1] * dims[2]);
  for (int i = 0; i < dims[0]; ++i) {
    real const x =
        boundsMin[0] + (boundsMax[0] - boundsMin[0]) * StaticCast<real>(i) / (dims[0] - 1);
    for (int j = 0; j < dims[1]; ++j) {
      real const y =
          boundsMin[1] + (boundsMax[1] - boundsMin[1]) * StaticCast<real>(j) / (dims[1] - 1);
      for (int k = 0; k < dims[2]; ++k) {
        real const z =
            boundsMin[2] + (boundsMax[2] - boundsMin[2]) * StaticCast<real>(k) / (dims[2] - 1);
        Real3 const d = {
            Abs(x - center[0]) - halfExtents[0],
            Abs(y - center[1]) - halfExtents[1],
            Abs(z - center[2]) - halfExtents[2]};
        Real3 const dPos = {Max(d[0], 0.0_r), Max(d[1], 0.0_r), Max(d[2], 0.0_r)};
        real const outside = Norm(dPos);
        real const inside = std::min(std::max({d[0], d[1], d[2]}), 0.0_r);
        int const idx = dims[1] * dims[2] * i + dims[2] * j + k;
        sdfData.values[idx] = outside + inside;
      }
    }
  }

  ExpectOK expectOK;
  MeshData const result = ReconstructSurfaceFromSdf(sdfData, expectOK);

  ASSERT_GT(result.GetNumNodes(), 0);

  // All vertices must lie inside the bounds (with small tolerance for surface tangent).
  real const tol = 0.05_r;
  Real3 vMin = {boundsMax[0], boundsMax[1], boundsMax[2]};
  Real3 vMax = {boundsMin[0], boundsMin[1], boundsMin[2]};
  for (int v = 0; v < result.GetNumNodes(); ++v) {
    Real3 const p = {
        result.coordinates[3 * v + 0],
        result.coordinates[3 * v + 1],
        result.coordinates[3 * v + 2]};
    EXPECT_GE(p[0], boundsMin[0] - tol);
    EXPECT_LE(p[0], boundsMax[0] + tol);
    EXPECT_GE(p[1], boundsMin[1] - tol);
    EXPECT_LE(p[1], boundsMax[1] + tol);
    EXPECT_GE(p[2], boundsMin[2] - tol);
    EXPECT_LE(p[2], boundsMax[2] + tol);
    for (int d = 0; d < 3; ++d) {
      vMin[d] = std::min(vMin[d], p[d]);
      vMax[d] = std::max(vMax[d], p[d]);
    }
  }

  // Box extents along each axis should match (within tolerance) so axis swaps fail.
  EXPECT_NEAR(static_cast<double>(vMax[0] - vMin[0]), 2.0 * 0.1, 0.1);
  EXPECT_NEAR(static_cast<double>(vMax[1] - vMin[1]), 2.0 * 0.2, 0.1);
  EXPECT_NEAR(static_cast<double>(vMax[2] - vMin[2]), 2.0 * 0.3, 0.1);
}

TEST(IsosurfaceReconstruction, RejectsNaNValues) {
  // Build a small valid SDF then inject a NaN into the values array.
  GridSdfData sdfData;
  sdfData.dims = {4, 4, 4};
  sdfData.values.resize(64);
  for (int i = 0; i < 64; ++i) {
    sdfData.values[i] = 1.0_r;
  }
  sdfData.values[10] = std::numeric_limits<real>::quiet_NaN();
  sdfData.bounds = Aabb(Real3{0.0_r, 0.0_r, 0.0_r}, Real3{1.0_r, 1.0_r, 1.0_r});

  ExpectNotOK expectNotOK;
  (void)ReconstructSurfaceFromSdf(sdfData, expectNotOK);
}
