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
#include <mochi_core/utils/decomposition_utils.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/vmatrix.h>

#include <algorithm>
#include <array>
#include <limits>

using namespace mochi;

/**************************************************************************************************
  AABB computation for meshes
*/

Aabb mochi::CalcAabb(Span<Real3 const> coordinates) {
  if (coordinates.empty()) {
    return Aabb{};
  }

  constexpr size_t kPointsPerBatch = 8; // Empirically selected for both AVX and NEON
  constexpr size_t kFloatsPerBatch = kPointsPerBatch * 3;
  using V = Simd<real, kPointsPerBatch>;
  using V3 = NdArray<V, 3>;
  using VFloats = Simd<real, kFloatsPerBatch>;

  Real3 min, max;
  size_t const count = coordinates.size();
  size_t i = 0;

  if (count >= kPointsPerBatch) {
    auto vmin = Load<VFloats>(&coordinates[0][0]);
    auto vmax = vmin;
    for (i = kPointsPerBatch; i + kPointsPerBatch <= count; i += kPointsPerBatch) {
      auto v = Load<VFloats>(&coordinates[i][0]);
      vmin = Min(vmin, v);
      vmax = Max(vmax, v);
    }
    // Deinterleave the batch and condense to Real3 min and max
    alignas(16) std::array<real, kFloatsPerBatch> minBuf MOCHI_NO_INIT;
    alignas(16) std::array<real, kFloatsPerBatch> maxBuf MOCHI_NO_INIT;
    Store(minBuf.data(), vmin);
    Store(maxBuf.data(), vmax);
    V3 minXYZ, maxXYZ;
    LoadTransposed(&minBuf[0], minXYZ[0], minXYZ[1], minXYZ[2]);
    LoadTransposed(&maxBuf[0], maxXYZ[0], maxXYZ[1], maxXYZ[2]);
    min = Real3{HMin(minXYZ[0]), HMin(minXYZ[1]), HMin(minXYZ[2])};
    max = Real3{HMax(maxXYZ[0]), HMax(maxXYZ[1]), HMax(maxXYZ[2])};
  } else {
    min = max = coordinates[0];
    i = 1;
  }

  // Let the compiler decide how to optimize the last few points.
  for (; i < count; ++i) {
    min = Min(min, coordinates[i]);
    max = Max(max, coordinates[i]);
  }

  return Aabb{min, max};
}

Aabb mochi::CalcAabbWithSortedIndices(
    Span<Real3 const> coordinates,
    Span<Real3 const> displacements,
    Span<int const> sortedIndices) {
  MOCHI_ASSERT(coordinates.size() == displacements.size());
  if (sortedIndices.empty()) {
    return Aabb{};
  }

  MOCHI_ASSERT_VERBOSE(
      std::ranges::is_sorted(sortedIndices) &&
          std::ranges::adjacent_find(sortedIndices) == sortedIndices.end(),
      "Expected indices in strictly increasing order.");

  Vec4r min = ToSimd(coordinates[sortedIndices[0]]) + ToSimd(displacements[sortedIndices[0]]);
  Vec4r max = min;

  size_t i = 1;
  for (; i < sortedIndices.size() - 1; ++i) {
    auto v = Load<Vec4r>(coordinates[sortedIndices[i]].data());
    v += Load<Vec4r>(displacements[sortedIndices[i]].data());
    min = Min(min, v);
    max = Max(max, v);
  }

  // Partial SIMD load for the last point
  if (i < sortedIndices.size()) {
    auto v = Load<3, Vec4r>(coordinates[sortedIndices[i]].data());
    v += Load<3, Vec4r>(displacements[sortedIndices[i]].data());
    min = Min(min, v);
    max = Max(max, v);
  }

  return Aabb{min, max};
}

Aabb mochi::CalcAabbWithDisplacements(
    Span<Real3 const> coordinates,
    Span<Real3 const> displacements) {
  MOCHI_ASSERT(coordinates.size() == displacements.size());
  if (coordinates.empty()) {
    return Aabb{};
  }

  constexpr size_t kPointsPerBatch = 8; // Empirically selected for both AVX and NEON
  constexpr size_t kFloatsPerBatch = kPointsPerBatch * 3;
  using V = Simd<real, kPointsPerBatch>;
  using V3 = NdArray<V, 3>;
  using VFloats = Simd<real, kFloatsPerBatch>;

  Real3 min, max;
  size_t const count = coordinates.size();
  size_t i = 0;

  if (count >= kPointsPerBatch) {
    auto vmin = Load<VFloats>(&coordinates[0][0]) + Load<VFloats>(&displacements[0][0]);
    auto vmax = vmin;
    for (i = kPointsPerBatch; i + kPointsPerBatch <= count; i += kPointsPerBatch) {
      auto v = Load<VFloats>(&coordinates[i][0]);
      auto d = Load<VFloats>(&displacements[i][0]);
      vmin = Min(vmin, v + d);
      vmax = Max(vmax, v + d);
    }
    // Condense the batch down to a Real3 min and max.
    alignas(16) std::array<real, kFloatsPerBatch> minBuf MOCHI_NO_INIT;
    alignas(16) std::array<real, kFloatsPerBatch> maxBuf MOCHI_NO_INIT;
    Store(&minBuf[0], vmin);
    Store(&maxBuf[0], vmax);
    V3 minXYZ, maxXYZ;
    LoadTransposed(&minBuf[0], minXYZ[0], minXYZ[1], minXYZ[2]);
    LoadTransposed(&maxBuf[0], maxXYZ[0], maxXYZ[1], maxXYZ[2]);
    min = Real3{HMin(minXYZ[0]), HMin(minXYZ[1]), HMin(minXYZ[2])};
    max = Real3{HMax(maxXYZ[0]), HMax(maxXYZ[1]), HMax(maxXYZ[2])};
  } else {
    min = max = coordinates[i] + displacements[i];
    i = 1;
  }

  // Let the compiler decide how to optimize the last few points.
  for (; i < count; ++i) {
    min = Min(min, coordinates[i] + displacements[i]);
    max = Max(max, coordinates[i] + displacements[i]);
  }

  return Aabb{min, max};
}

/**************************************************************************************************
  OBB computation for meshes
*/

Obb mochi::CalcObb(Span<Real3 const> coordinates) {
  if (coordinates.empty()) {
    return Obb{};
  }

  int const numCoords = isize(coordinates);
  int const numCoordsMinusOne = numCoords - 1;
  int i = 0;

  // Compute mean of point cloud.
  auto mean = ToSimd(coordinates[0]);
  for (i = 1; i < numCoordsMinusOne; ++i) {
    mean += Load<Vec4r>(coordinates[i].data());
  }
  if (i < numCoords) {
    mean += ToSimd(coordinates[i]);
  }
  mean /= static_cast<real>(numCoords);

  // Compute covariance matrix.
  VSymMatrix3x3r cov = {SimdZero(), SimdZero()};
  for (i = 0; i < numCoordsMinusOne; ++i) {
    auto offs = Load<Vec4r>(coordinates[i].data()) - mean;
    cov += OuterSym3(offs, offs);
  }
  if (i < numCoords) {
    auto offs = Load<3, Vec4r>(coordinates[i].data()) - mean;
    cov += OuterSym3(offs, offs);
  }

  // Perform eigenanalysis of covariance matrix to find directions of maximum spread.
  // The eigenvectors will become the rotation of our Obb.
  Vec4r unused;
  VMatrix3x3r rotation;
  AnalyticalEigendecompSym3x3(cov, unused, &rotation);
  rotation = Transpose3x3(rotation);

  // Ensure only rotation is encoded in the matrix.
  if (Det3x3(rotation) < 0.0_r) {
    rotation = -rotation;
  }

  // Determine minimum and maximum extents.
  // Expand slightly to ensure point cloud use to create the Obb is contained by it.
  Vec4r minExtents = DotVecMat3x3(ToSimd(coordinates[0]), rotation);
  Vec4r maxExtents = minExtents;

  for (i = 1; i < numCoordsMinusOne; ++i) {
    auto coord = DotVecMat3x3(Load<Vec4r>(coordinates[i].data()), rotation);
    minExtents = Min(minExtents, coord);
    maxExtents = Max(maxExtents, coord);
  }
  if (i < numCoords) {
    auto coord = DotVecMat3x3(Load<3, Vec4r>(coordinates[i].data()), rotation);
    minExtents = Min(minExtents, coord);
    maxExtents = Max(maxExtents, coord);
  }

  real constexpr kEpsilon = 1e1_r * std::numeric_limits<real>::epsilon();
  Vec4r translation = DotMatVec3x3(rotation, (minExtents + maxExtents) * 0.5_r);
  Vec4r extents = ((maxExtents - minExtents) * 0.5_r) + kEpsilon;
  return Obb{MatrixTransformRT{rotation, translation}, extents};
}

Obb mochi::CalcObb(Span<Vec4r const> coordinates) {
  if (coordinates.empty()) {
    return Obb{};
  }

  int const numCoords = isize(coordinates);

  // Compute mean of point cloud.
  Vec4r mean = coordinates[0];
  for (int i = 1; i < numCoords; ++i) {
    mean += coordinates[i];
  }
  mean /= static_cast<real>(numCoords);

  // Compute covariance matrix.
  VSymMatrix3x3r cov = {SimdZero(), SimdZero()};
  for (int i = 0; i < numCoords; ++i) {
    auto offs = coordinates[i] - mean;
    cov += OuterSym3(offs, offs);
  }

  // Perform eigenanalysis of covariance matrix to find directions of maximum spread.
  // The eigenvectors will become the rotation of our Obb.
  Vec4r unused;
  VMatrix3x3r rotation;
  AnalyticalEigendecompSym3x3(cov, unused, &rotation);
  rotation = Transpose3x3(rotation);

  // Ensure only rotation is encoded in the matrix.
  if (Det3x3(rotation) < 0.0_r) {
    rotation = -rotation;
  }

  // Determine minimum and maximum extents.
  Vec4r minExtents = DotVecMat3x3(coordinates[0], rotation);
  Vec4r maxExtents = minExtents;

  for (int i = 1; i < numCoords; ++i) {
    auto coord = DotVecMat3x3(coordinates[i], rotation);
    minExtents = Min(minExtents, coord);
    maxExtents = Max(maxExtents, coord);
  }

  real constexpr kEpsilon = 1e1_r * std::numeric_limits<real>::epsilon();
  Vec4r translation = DotMatVec3x3(rotation, (minExtents + maxExtents) * 0.5_r);
  Vec4r extents = ((maxExtents - minExtents) * 0.5_r) + kEpsilon;
  return Obb{MatrixTransformRT{rotation, translation}, extents};
}

/*************************************************************************************************
  Closest face computation for meshes
*/

int mochi::ClosestFacePointTriangularMesh(
    Real3 const& p,
    Span<Real3 const> nodes,
    Span<Int3 const> faces) {
  Vec4r vp = ToSimd(p);
  int nearestFace = -1;
  real nearestDist = std::numeric_limits<real>::infinity();

  VDistanceSignParams signParams;
  signParams.computeSign = false;

  for (size_t i = 0; i < faces.size(); ++i) {
    // Get face geometry
    Int3 const& face = faces[i];
    Vec4r A = ToSimd(nodes[face[0]]);
    Vec4r B = ToSimd(nodes[face[1]]);
    Vec4r C = ToSimd(nodes[face[2]]);
    real dist = Get0(VDistancePointTriangle(vp, A, B, C, signParams, nullptr));
    if (dist < nearestDist) {
      nearestDist = dist;
      nearestFace = static_cast<int>(i);
    }
  }

  return nearestFace;
}
