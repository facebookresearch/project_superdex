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
#include <mochi_core/memory/filo_allocator.h>
#include <mochi_core/utils/decomposition_utils.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/rand_utils.h>
#include <mochi_core/utils/vmatrix.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <numeric>
#include <type_traits>

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
    Span<int const> sortedIndices) {
  if (sortedIndices.empty()) {
    return Aabb{};
  }

#if MOCHI_ASSERT_VERBOSE_ENABLED
  for (size_t i = 1; i < sortedIndices.size(); ++i) {
    MOCHI_ASSERT_VERBOSE(
        sortedIndices[i] > sortedIndices[i - 1], "Expected indices in increasing order.");
  }
#endif // MOCHI_ASSERT_VERBOSE_ENABLED

  Vec4r min = ToSimd(coordinates[sortedIndices[0]]);
  Vec4r max = min;

  size_t i = 1;
  for (; i < sortedIndices.size() - 1; ++i) {
    auto v = Load<Vec4r>(coordinates[sortedIndices[i]].data());
    min = Min(min, v);
    max = Max(max, v);
  }

  // Partial SIMD load for the last point
  if (i < sortedIndices.size()) {
    auto v = Load<3, Vec4r>(coordinates[sortedIndices[i]].data());
    min = Min(min, v);
    max = Max(max, v);
  }

  return Aabb{min, max};
}

Aabb mochi::CalcAabbWithDisplacementsAndSortedIndices(
    Span<Real3 const> coordinates,
    Span<Real3 const> displacements,
    Span<int const> sortedIndices) {
  MOCHI_ASSERT(coordinates.size() == displacements.size());
  if (sortedIndices.empty()) {
    return Aabb{};
  }

#if MOCHI_ASSERT_VERBOSE_ENABLED
  for (size_t i = 1; i < sortedIndices.size(); ++i) {
    MOCHI_ASSERT_VERBOSE(
        sortedIndices[i] > sortedIndices[i - 1], "Expected indices in increasing order.");
  }
#endif // MOCHI_ASSERT_VERBOSE_ENABLED

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

/**************************************************************************************************
  Bounding sphere computation
*/

// Compute a midpoint without overflowing large values or losing equal subnormal values.
[[nodiscard]] static Vec4r CalcMidpoint(Vec4r a, Vec4r b) {
  return Vec4r{
      std::midpoint(Get<0>(a), Get<0>(b)),
      std::midpoint(Get<1>(a), Get<1>(b)),
      std::midpoint(Get<2>(a), Get<2>(b)),
      std::midpoint(Get<3>(a), Get<3>(b))};
}

// Measure the farthest AABB corner from the rounded center, which need not be the exact midpoint.
[[nodiscard]] static real CalcAabbRadiusSqr(Vec4r min, Vec4r max, Vec4r center) {
  Vec4r const farthestOffset = Max(Abs(min - center), Abs(max - center));
  return NormSqr<3>(farthestOffset);
}

// Center on the AABB; Fast uses the farthest input point and Fastest the farthest AABB corner.
static Sphere CalcBoundingSphere_FromAabb(Span<Real3 const> coordinates, bool shrinkFromCenter) {
  if (coordinates.empty()) {
    return {};
  }

  using V = Simd<real>;
  using V3 = NdArray<V, 3>;
  size_t const count = coordinates.size();

  Aabb const aabb = CalcAabb(coordinates);
  Vec4r const center = CalcMidpoint(aabb.VGetMin(), aabb.VGetMax());
  real radiusSqr = 0_r;

  if (shrinkFromCenter) {
    size_t i = 0;
    V maxDistSqr = {};
    V3 pt, vCenter = BroadcastEach<V>(ToReal3(center));
    for (; i + V::kSize <= count; i += V::kSize) {
      LoadTransposed(&coordinates[i][0], pt);
      maxDistSqr = Max(maxDistSqr, NormSqr(pt - vCenter));
    }

    // Pad the tail to one SIMD batch and mask its unused lanes.
    alignas(V) Real3 buf[V::kSize] = {};
    std::copy(coordinates.begin() + i, coordinates.end(), buf);
    LoadTransposed(&buf[0][0], pt);
    using I = std::conditional_t<sizeof(real) == 8, int64_t, int>;
    using VI = Simd<I, V::kSize>;
    I const numRemaining = static_cast<I>(count - i);
    auto const mask = Sequence<VI>() < numRemaining;
    maxDistSqr = Max(maxDistSqr, Select(mask, NormSqr(pt - vCenter), V{}));

    radiusSqr = HMax(maxDistSqr);
  } else {
    radiusSqr = CalcAabbRadiusSqr(aabb.VGetMin(), aabb.VGetMax(), center);
  }

  // Ensure both distance and squared-distance tests keep boundary points inside after rounding.
  real constexpr kPadding = 4_r * std::numeric_limits<real>::epsilon();
  radiusSqr += radiusSqr * kPadding;

  real const radius = Sqrt(radiusSqr);
  return Sphere{center, radius};
}

// Indexed counterpart of CalcBoundingSphere_FromAabb; only selected coordinates affect the sphere.
static Sphere CalcBoundingSphereIndexed_FromAabb(
    Span<Real3 const> coordinates,
    Span<int const> indices,
    bool shrinkFromCenter) {
  if (indices.empty()) {
    return {};
  }

  Vec4r min, max;
  min = max = ToSimd(coordinates[indices[0]]);
  for (size_t i = 1; i < indices.size(); ++i) {
    Vec4r pt = ToSimd(coordinates[indices[i]]);
    min = Min(min, pt);
    max = Max(max, pt);
  }

  Vec4r const center = CalcMidpoint(min, max);
  real radiusSqr = 0_r;

  if (shrinkFromCenter) {
    for (int idx : indices) {
      Vec4r pt = ToSimd(coordinates[idx]);
      radiusSqr = Max(radiusSqr, NormSqr(pt - center));
    }
  } else {
    radiusSqr = CalcAabbRadiusSqr(min, max, center);
  }

  // Ensure both distance and squared-distance tests keep boundary points inside after rounding.
  real constexpr kPadding = 4_r * std::numeric_limits<real>::epsilon();
  radiusSqr += radiusSqr * kPadding;

  real const radius = Sqrt(radiusSqr);
  return Sphere{ToReal3(center), radius};
}

namespace {

// Best applies Welzl's incremental algorithm to deterministically shuffled, normalized points.
// Recursion adds at most four boundary constraints; intermediate spheres store squared radii.
real constexpr kBoundingSphereTolerance = 16_r * std::numeric_limits<real>::epsilon();

struct SphereSqr {
  Vec4r center{};
  real radiusSqr{-1_r}; // A negative radius represents no sphere.

  [[nodiscard]] bool IsValid() const {
    return radiusSqr >= 0_r;
  }

  [[nodiscard]] bool ContainsPoint(Vec4r point) const {
    if (!IsValid()) {
      return false;
    }
    real const distSqr = NormSqr<3>(point - center);
    // Avoid promoting a nominal boundary point because its distance rounded a few ulps upward.
    return distSqr <= radiusSqr + kBoundingSphereTolerance * Max(distSqr, radiusSqr);
  }
};

} // namespace

// Construct the sphere supported by one boundary point.
[[nodiscard]] static SphereSqr SphereSqrFromPoint(Vec4r point) {
  return {point, 0_r};
}

// Construct the diameter sphere supported by two boundary points.
[[nodiscard]] static SphereSqr SphereSqrFromTwoPoints(Vec4r a, Vec4r b) {
  Vec4r const center = a + (b - a) * 0.5_r;
  real const radiusSqr = Max(NormSqr<3>(a - center), NormSqr<3>(b - center));
  return {center, radiusSqr};
}

// Construct the circumcircle of three non-collinear boundary points in their plane.
[[nodiscard]] static SphereSqr SphereSqrFromThreePoints(Vec4r a, Vec4r b, Vec4r c) {
  Vec4r const ab = b - a;
  Vec4r const ac = c - a;
  real const abSqr = NormSqr<3>(ab);
  real const acSqr = NormSqr<3>(ac);
  Vec4r const normal = Cross3(ab, ac);
  real const normalSqr = NormSqr<3>(normal);
  real const denominator = 2_r * normalSqr;
  if (denominator == 0_r) {
    return {};
  }

  Vec4r const offset = (abSqr * Cross3(ac, normal) + acSqr * Cross3(normal, ab)) / denominator;
  Vec4r const center = a + offset;
  real const radiusSqr =
      Max(NormSqr<3>(a - center), Max(NormSqr<3>(b - center), NormSqr<3>(c - center)));
  if (!IsFinite(center) || !IsFinite(radiusSqr)) {
    return {};
  }
  return {center, radiusSqr};
}

// Construct the circumsphere of four affinely independent boundary points.
[[nodiscard]] static SphereSqr SphereSqrFromFourPoints(Vec4r a, Vec4r b, Vec4r c, Vec4r d) {
  Vec4r const ab = b - a;
  Vec4r const ac = c - a;
  Vec4r const ad = d - a;
  real const abSqr = NormSqr<3>(ab);
  real const acSqr = NormSqr<3>(ac);
  real const adSqr = NormSqr<3>(ad);
  Vec4r const acCrossAd = Cross3(ac, ad);
  Vec4r const adCrossAb = Cross3(ad, ab);
  Vec4r const abCrossAc = Cross3(ab, ac);
  real const denominator = 2_r * Dot<3>(ab, acCrossAd);
  if (denominator == 0_r) {
    return {};
  }

  Vec4r const offset = (abSqr * acCrossAd + acSqr * adCrossAb + adSqr * abCrossAc) / denominator;
  Vec4r const center = a + offset;
  real const radiusSqr =
      Max(Max(NormSqr<3>(a - center), NormSqr<3>(b - center)),
          Max(NormSqr<3>(c - center), NormSqr<3>(d - center)));
  if (!IsFinite(center) || !IsFinite(radiusSqr)) {
    return {};
  }
  return {center, radiusSqr};
}

// Keep a containing candidate, expanding its radius to absorb any tolerated boundary roundoff.
static void
TrySmallerContainingSphere(SphereSqr candidate, Span<Vec4r const> points, SphereSqr& smallest) {
  if (!candidate.IsValid()) {
    return;
  }
  real maxRadiusSqr = candidate.radiusSqr;
  for (Vec4r point : points) {
    real const distSqr = NormSqr<3>(point - candidate.center);
    if (distSqr >
        candidate.radiusSqr + kBoundingSphereTolerance * Max(distSqr, candidate.radiusSqr)) {
      return;
    }
    maxRadiusSqr = Max(maxRadiusSqr, distSqr);
  }
  candidate.radiusSqr = maxRadiusSqr;
  if (!smallest.IsValid() || candidate.radiusSqr < smallest.radiusSqr) {
    smallest = candidate;
  }
}

// A dependent support is bounded by a subset of at most three points; enumerate those subsets.
[[nodiscard]] static SphereSqr CalcDegenerateSupportSphere(Span<Vec4r const> points) {
  SphereSqr smallest;
  for (size_t i = 0; i < points.size(); ++i) {
    TrySmallerContainingSphere(SphereSqrFromPoint(points[i]), points, smallest);
    for (size_t j = i + 1; j < points.size(); ++j) {
      TrySmallerContainingSphere(SphereSqrFromTwoPoints(points[i], points[j]), points, smallest);
      for (size_t k = j + 1; k < points.size(); ++k) {
        TrySmallerContainingSphere(
            SphereSqrFromThreePoints(points[i], points[j], points[k]), points, smallest);
      }
    }
  }
  MOCHI_ASSERT_VERBOSE(smallest.IsValid(), "Failed to bound a degenerate support set.");
  return smallest;
}

// Construct the sphere fixed by Welzl's boundary support, with a fallback for dependent points.
[[nodiscard]] static SphereSqr CalcSupportSphere(Span<Vec4r const> support) {
  switch (support.size()) {
    case 0:
      return {};
    case 1:
      return SphereSqrFromPoint(support[0]);
    case 2:
      return SphereSqrFromTwoPoints(support[0], support[1]);
    case 3: {
      SphereSqr const sphere = SphereSqrFromThreePoints(support[0], support[1], support[2]);
      return sphere.IsValid() ? sphere : CalcDegenerateSupportSphere(support);
    }
    case 4: {
      SphereSqr const sphere =
          SphereSqrFromFourPoints(support[0], support[1], support[2], support[3]);
      return sphere.IsValid() ? sphere : CalcDegenerateSupportSphere(support);
    }
  }
  MOCHI_ASSERT(false, "A 3D bounding sphere has at most four support points.");
  return {};
}

// Map a point into the centered, uniformly scaled coordinate system used by the solver.
[[nodiscard]] static Vec4r NormalizePoint(Vec4r point, Vec4r origin, real invScale) {
  return (point - origin) * invScale;
}

// Scan in SIMD batches, then rescan a flagged batch to preserve the first outside point's index.
[[nodiscard]] static size_t FindFirstPointOutsideSphere(
    Span<Real3 const> points,
    size_t begin,
    size_t end,
    SphereSqr const& sphere,
    Vec4r origin,
    real invScale) {
  if (!sphere.IsValid()) {
    return begin;
  }

  using V = Simd<real>;
  using V3 = NdArray<V, 3>;

  V3 const vOrigin = BroadcastEach<V>(ToReal3(origin));
  V3 const vCenter = BroadcastEach<V>(ToReal3(sphere.center));
  V const vInvScale = invScale;
  V const vRadiusSqr = sphere.radiusSqr;
  V const vTolerance = kBoundingSphereTolerance;

  size_t i = begin;
  for (; i + V::kSize <= end; i += V::kSize) {
    V3 point;
    LoadTransposed(&points[i][0], point);
    V const normalizedDistSqr = NormSqr((point - vOrigin) * vInvScale - vCenter);
    auto const outside =
        normalizedDistSqr > vRadiusSqr + vTolerance * Max(normalizedDistSqr, vRadiusSqr);
    if (AnyTrue(outside)) {
      for (size_t j = i; j < i + V::kSize; ++j) {
        if (!sphere.ContainsPoint(NormalizePoint(ToSimd(points[j]), origin, invScale))) {
          return j;
        }
      }
    }
  }

  for (; i < end; ++i) {
    if (!sphere.ContainsPoint(NormalizePoint(ToSimd(points[i]), origin, invScale))) {
      return i;
    }
  }
  return end;
}

// Process a prefix incrementally. Each outside point becomes fixed boundary support for a
// recursive solve of the preceding prefix.
[[nodiscard]] static SphereSqr CalcBoundingSphereWelzl(
    Span<Real3 const> points,
    size_t pointCount,
    std::array<Vec4r, 4>& support,
    size_t supportCount,
    Vec4r origin,
    real invScale) {
  // Each recursive call fixes one more support point, so recursion depth is at most four.
  SphereSqr sphere = CalcSupportSphere(Span<Vec4r const>{support.data(), supportCount});
  if (supportCount == support.size()) {
    return sphere;
  }

  size_t i = 0;
  while (i < pointCount) {
    i = FindFirstPointOutsideSphere(points, i, pointCount, sphere, origin, invScale);
    if (i == pointCount) {
      break;
    }
    support[supportCount] = NormalizePoint(ToSimd(points[i]), origin, invScale);
    sphere = CalcBoundingSphereWelzl(points, i, support, supportCount + 1, origin, invScale);
    ++i;
  }
  return sphere;
}

// A reproducible shuffle avoids common order-dependent worst cases without shared RNG.
static void DeterministicallyShuffle(Span<Real3> points) {
  constexpr uint32_t kSeed = 0x9E3779B9u;
  uint64_t const pointCount = points.size();
  uint32_t const sizeHash =
      static_cast<uint32_t>(pointCount) ^ static_cast<uint32_t>(pointCount >> 32);
  auto random = XorShift32Generator(kSeed ^ sizeHash);
  for (size_t i = points.size(); i > 1; --i) {
    uint64_t const randomValue =
        (static_cast<uint64_t>(random()) << 32) | static_cast<uint64_t>(random());
    std::swap(points[i - 1], points[static_cast<size_t>(randomValue % i)]);
  }
}

// Find the farthest input from the final, denormalized center.
[[nodiscard]] static real CalcMaxDistanceSqr(Span<Real3 const> points, Vec4r center) {
  using V = Simd<real>;
  using V3 = NdArray<V, 3>;

  V3 const vCenter = BroadcastEach<V>(ToReal3(center));
  V maxDistSqr{};
  size_t i = 0;
  for (; i + V::kSize <= points.size(); i += V::kSize) {
    V3 point;
    LoadTransposed(&points[i][0], point);
    maxDistSqr = Max(maxDistSqr, NormSqr(point - vCenter));
  }

  real result = HMax(maxDistSqr);
  for (; i < points.size(); ++i) {
    result = Max(result, NormSqr<3>(ToSimd(points[i]) - center));
  }
  return result;
}

// Solve in an AABB-centered, uniformly scaled frame, then recompute the radius in the input frame.
[[nodiscard]] static Sphere CalcBoundingSphereBestInPlace(Span<Real3> points) {
  Aabb const aabb = CalcAabb(points);
  Vec4r const min = aabb.VGetMin();
  Vec4r const max = aabb.VGetMax();
  Vec4r const origin = CalcMidpoint(min, max);
  real const scale = HMax<3>(Max(Abs(min - origin), Abs(max - origin)));
  if (scale == 0_r) {
    return Sphere{points[0], 0_r};
  }

  real const normalizationScale = Max(scale, std::numeric_limits<real>::min());
  real const invNormalizationScale = 1_r / normalizationScale;
  DeterministicallyShuffle(points);

  std::array<Vec4r, 4> support{};
  SphereSqr const normalizedSphere =
      CalcBoundingSphereWelzl(points, points.size(), support, 0, origin, invNormalizationScale);
  MOCHI_ASSERT_VERBOSE(normalizedSphere.IsValid(), "Failed to compute a bounding sphere.");

  // Recomputing around the rounded, denormalized center makes the public sphere conservative.
  Vec4r const center = origin + normalizedSphere.center * normalizationScale;
  real radiusSqr = CalcMaxDistanceSqr(points, center);
  real constexpr kPadding = 4_r * std::numeric_limits<real>::epsilon();
  radiusSqr += radiusSqr * kPadding;
  return Sphere{center, Sqrt(radiusSqr)};
}

constexpr size_t kBoundingSphereStackBytes = 16 * 1024;

// Copy the points because the incremental solver shuffles its input.
[[nodiscard]] static Sphere CalcBoundingSphereBest(Span<Real3 const> coordinates) {
  if (coordinates.empty()) {
    return {};
  }

  MOCHI_FILO_STACK_ALLOCATOR(allocator, kBoundingSphereStackBytes);
  DynamicArray<Real3> points(&allocator);
  points.resize_noinit(coordinates.size());
  std::copy(coordinates.begin(), coordinates.end(), points.begin());
  return CalcBoundingSphereBestInPlace(points);
}

// Gather the selected points into mutable scratch space for the shared in-place solver.
[[nodiscard]] static Sphere CalcBoundingSphereBestIndexed(
    Span<Real3 const> coordinates,
    Span<int const> indices) {
  if (indices.empty()) {
    return {};
  }

  MOCHI_FILO_STACK_ALLOCATOR(allocator, kBoundingSphereStackBytes);
  DynamicArray<Real3> points(&allocator);
  points.resize_noinit(indices.size());
  for (size_t i = 0; i < indices.size(); ++i) {
    points[i] = coordinates[indices[i]];
  }
  return CalcBoundingSphereBestInPlace(points);
}

Sphere mochi::CalcBoundingSphere(Span<Real3 const> coordinates, BoundingSphereAlgorithm algorithm) {
  static_assert(
      static_cast<int>(BoundingSphereAlgorithm::Count) == 3,
      "Please update this code if you add a new algorithm.");
  switch (algorithm) {
    case BoundingSphereAlgorithm::Fastest:
      return CalcBoundingSphere_FromAabb(coordinates, /*shrinkFromCenter*/ false);

    case BoundingSphereAlgorithm::Fast:
      return CalcBoundingSphere_FromAabb(coordinates, /*shrinkFromCenter*/ true);

    case BoundingSphereAlgorithm::Best:
      return CalcBoundingSphereBest(coordinates);

    case BoundingSphereAlgorithm::Count:
      break;
  }
  MOCHI_ASSERT(false, "Invalid BoundingSphereAlgorithm");
  return {};
}

Sphere mochi::CalcBoundingSphereIndexed(
    Span<Real3 const> coordinates,
    Span<int const> indices,
    BoundingSphereAlgorithm algorithm) {
  static_assert(
      static_cast<int>(BoundingSphereAlgorithm::Count) == 3,
      "Please update this code if you add a new algorithm.");
  switch (algorithm) {
    case BoundingSphereAlgorithm::Fastest:
      return CalcBoundingSphereIndexed_FromAabb(coordinates, indices, /*shrinkFromCenter*/ false);

    case BoundingSphereAlgorithm::Fast:
      return CalcBoundingSphereIndexed_FromAabb(coordinates, indices, /*shrinkFromCenter*/ true);

    case BoundingSphereAlgorithm::Best:
      return CalcBoundingSphereBestIndexed(coordinates, indices);

    case BoundingSphereAlgorithm::Count:
      break;
  }
  MOCHI_ASSERT(false, "Invalid BoundingSphereAlgorithm");
  return {};
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
