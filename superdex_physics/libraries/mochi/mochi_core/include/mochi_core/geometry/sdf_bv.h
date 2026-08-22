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

#pragma once

#include <mochi_core/geometry/grid_sdf.h>
#include <mochi_core/geometry/scalar_field.h>
#include <mochi_core/geometry/sphere.h>
#include <mochi_core/mochi_config.h>
#include <mochi_core/mochi_platform.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/concepts.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/span.h>
#include <mochi_core/utils/vmatrix.h>

namespace mochi {

/** @brief Represents a bounding volume defined by the level set of a signed distance field (SDF).
 */
struct SdfBv {
  // Pointer to the SDF.
  GridSdf const* gridSdf = nullptr;

  // Distance threshold that defines the level-set of the SDF. The distance is expressed in Points
  // space (whose scale may be different than Grid space). Positive values expand the volume.
  // Negative values contract the volume.
  real distanceThreshold;

  // Transform matrix that converts from Points space to Grid space. It may encode rotation,
  // translation and uniform scaling.
  VMatrix4x4r gridFromPointsT;
};

// NOTE: HasOverlap(SdfBv, ShapeT) is currently only supported for ShapeT = Sphere.
MOCHI_FORCE_INLINE bool HasOverlap(SdfBv const& sdfBv, Sphere const& sphere) {
  Real3 centerInGrid = ToReal3(DotVecMat4x4(sphere.VGetCenter(), sdfBv.gridFromPointsT));
  real outDistance;

  constexpr auto kSamplerOptions = TrilinearSamplerOptions<GridExtrapolation::LowerBound>{};
  sdfBv.gridSdf->GetDistanceGrid().TrilinearSample(
      MakeSingletonConstSpan(centerInGrid), MakeSingletonSpan(outDistance), kSamplerOptions);

  return outDistance * sdfBv.gridSdf->GetActorFromGridScale() <=
      sphere.GetRadius() + sdfBv.distanceThreshold;
}

template <typename ShapeT, MOCHI_CONCEPT(IsPrimitiveShape<ShapeT>)>
MOCHI_FORCE_INLINE bool HasOverlap(ShapeT const& shape, SdfBv const& sdfBv) {
  return HasOverlap(sdfBv, shape);
}

/**
 * @brief Checks for overlap between an SDF bounding volume and multiple spheres in batch.
 *
 * @tparam kMaxBatchSize Maximum number of spheres to process in a batch.
 *
 * @param batchSize Number of spheres to process in this batch (must not exceed kMaxBatchSize).
 * @param sdfBv The SDF bounding volume to check against all spheres.
 * @param spheres Span of spheres to check for overlap with the SDF bounding volume.
 * @param outHasOverlap Output span to store overlap results (must be at least batchSize in length).
 *
 * @note HasOverlapBatch(SdfBv, ShapeT) is currently only supported for ShapeT = Sphere.
 */
template <int kMaxBatchSize>
void HasOverlapBatch(
    int batchSize,
    SdfBv const& sdfBv,
    Span<Sphere const> spheres,
    Span<bool> outHasOverlap) {
  MOCHI_ASSERT_VERBOSE(
      (batchSize >= 0) && (batchSize <= Min(kMaxBatchSize, isize(spheres), isize(outHasOverlap))),
      "Invalid batch size.");
  Real3 centersInGrid[kMaxBatchSize] MOCHI_NO_INIT;
  for (int i = 0; i < batchSize; ++i) {
    centersInGrid[i] = ToReal3(DotVecMat4x4(spheres[i].VGetCenter(), sdfBv.gridFromPointsT));
  }
  real outDistance[kMaxBatchSize] MOCHI_NO_INIT;

  constexpr auto kSamplerOptions = TrilinearSamplerOptions<GridExtrapolation::LowerBound>{};
  sdfBv.gridSdf->GetDistanceGrid().TrilinearSample(
      Span(&centersInGrid[0], batchSize), Span(&outDistance[0], batchSize), kSamplerOptions);

  real const actorFromGridScale = sdfBv.gridSdf->GetActorFromGridScale();
  for (int i = 0; i < batchSize; ++i) {
    outHasOverlap[i] =
        (outDistance[i] * actorFromGridScale <= spheres[i].GetRadius() + sdfBv.distanceThreshold);
  }
}

namespace details {
template <>
inline constexpr bool IsSdfBvDef<SdfBv> = true;
} // namespace details

} // namespace mochi
