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

// PLEASE DO NOT ADD OTHER INCLUDES HERE. This header is included in the mochi_physics public API.
#include <mochi_core/contact/contact_params.h>
#include <mochi_core/mochi_platform.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/reflection.h>

namespace mochi {

/**
 * @brief The maximum voxel size for an SDF will be determined by one of these measurements scaled
 * by @ref GridSdfParams::resolutionDelta.
 */
enum struct GridSdfResolutionMode {
  /**
   * @brief Use the largest dimension of the mesh's axis-aligned bounding box (AABB) as reference.
   */
  LargestAxis,

  /** @brief Use the smallest dimension of the mesh's AABB as reference. */
  SmallestAxis,

  /** @brief Use the average of all three AABB dimensions as reference. */
  MeanAxis,

  /** @brief Use the largest edge length in the mesh as reference. */
  LargestEdge,

  /** @brief Use the smallest edge length in the mesh as reference. */
  SmallestEdge,

  /** @brief Use the average edge length in the mesh as reference. */
  MeanEdge,

  /**
   * @brief Use @ref GridSdfParams::resolutionDelta to set the maximum voxel size in meters (no
   * scaling based on the features of the mesh nor AABB).
   */
  Explicit,

  /** @brief Total number of resolution modes. */
  Count
};

} // namespace mochi

MOCHI_ENUM_BEGIN(mochi::GridSdfResolutionMode)
MOCHI_ENUM_ITEM(LargestAxis);
MOCHI_ENUM_ITEM(SmallestAxis);
MOCHI_ENUM_ITEM(MeanAxis);
MOCHI_ENUM_ITEM(LargestEdge);
MOCHI_ENUM_ITEM(SmallestEdge);
MOCHI_ENUM_ITEM(MeanEdge);
MOCHI_ENUM_ITEM(Explicit);
MOCHI_ENUM_COUNT(Count);
MOCHI_ENUM_END()

namespace mochi {

/**
 * @brief Default value for @ref GridSdfParams::boundaryPaddingDist [m]
 *
 * @note For best performance, an SDF should have more padding than the actor's contact penalty
 * threshold distance. This default was chosen to be 5X the default penalty threshold distance so
 * that an SDF will have sufficient padding when loaded with any scale value greater than or equal
 * to 0.2 (per axis).
 *
 * @see ContactParams::penaltyThresholdDefault
 */
static constexpr real kGridSdfDefaultBoundaryPadding =
    5_r * ContactParams{}.penaltyThresholdDefault;

/**
 * @brief Parameters controlling the resolution of grid-based Signed Distance Fields (SDF).
 *
 * @details A GridSdf represents the SDF to a closed surface mesh as a 3D grid where each vertex
 * stores the signed distance to the surface mesh. The SDF grid resolution and bounds are controlled
 * by these parameters.
 */
struct GridSdfParams {
  /**
   * @brief Defines the mesh feature used as reference measurement for computing the voxel size.
   *
   * @see resolutionDelta
   */
  GridSdfResolutionMode resolutionMode = GridSdfResolutionMode::MeanEdge;

  /**
   * @brief The maximum voxel size will be @ref resolutionDelta times the reference measurement
   * computed using @ref resolutionMode.
   *
   * @note The actual voxel size may be smaller because the number of grid cells will be rounded up
   * to an integer value and clamped to a minimum of @ref minGridResolution.
   * @note Increasing @ref resolutionDelta will result in larger voxels. Doing so will make it
   * harder for the SDF to resolve fine details of the mesh, but it will save memory and improve SDF
   * generation speed.
   * @note Decreasing @ref resolutionDelta will improve the SDF's ability to resolve fine details,
   * but it will take more memory and more time to generate. This cost scales proportional to N^3.
   * Use with care.
   *
   * @see resolutionMode
   */
  Real3 resolutionDelta = {0.25_r, 0.25_r, 0.25_r};

  /**
   * @brief Additional distance (in meters) to expand the mesh's axis-aligned bounding box (AABB)
   * when determining the SDF grid bounds.
   *
   * @details The SDF grid will cover the mesh's AABB expanded by this distance in all directions.
   * This ensures the SDF has valid distance values even slightly outside the mesh's AABB.
   *
   * @note In practice, this value should be greater than the value returned by @ref
   * ContactParams::GetPenaltyThresholdDist. If the SDF did not have enough padding, then Mochi
   * would have to use a less efficient algorithm for collision detection, but it would still work.
   * @note When you load a model and bake a uniform scale value less than 1.0, the model's
   * pre-computed SDF will also be scaled, resulting in less absolute padding. The default padding
   * value is larger than the default contact penalty threshold distance for this reason.
   *
   * @see kGridSdfDefaultBoundaryPadding
   */
  real boundaryPaddingDist = kGridSdfDefaultBoundaryPadding;

  /**
   * @brief The minimum grid resolution measured in number of voxels per axis.
   *
   * @note This may override the voxel count (and voxel size) computed using @ref resolutionMode and
   * @ref resolutionDelta.
   */
  Int3 minGridResolution = {6, 6, 6};

#if MOCHI_LANGUAGE_CPP20
  bool operator==(GridSdfParams const&) const = default;
#endif

  // clang-format off
  MOCHI_STRUCT_BEGIN(mochi::GridSdfParams)
  MOCHI_FIELD(resolutionMode)
  MOCHI_FIELD(resolutionDelta)
  MOCHI_FIELD(boundaryPaddingDist) MOCHI_ATTRIBUTE(Units("m"))
  MOCHI_FIELD(minGridResolution)
  MOCHI_STRUCT_END()
  // clang-format on
};

} // namespace mochi
