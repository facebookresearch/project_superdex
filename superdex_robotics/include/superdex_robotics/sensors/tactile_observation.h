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

#include <superdex_robotics/sensors/sensor_format.h>

#include <superdex_physics.h>

namespace superdex::robotics {

/* @brief Geometry used to flatten a tactile spherical cap into a rectangle. */
enum class TactileRectProjectionType {
  AzimuthalEqualArea = 1,
  AzimuthalEquidistant = 2,
  Octahedral = 3,
};

/* @brief Resampling from projected taxels into the output grid. */
enum class TactileRectRasterizationType {
  /* Distribute every taxel's force across its discrete projected Voronoi region. */
  ConservativeVoronoi = 1,

  /* Accumulate every taxel's complete force into its nearest output cell. */
  NearestCell = 2,
};

/* @brief Handling for source taxels outside a configured tactile spherical cap. */
enum class TactileRectOutsidePolicy {
  Reject = 1,
  Exclude = 2,
};

/* @brief Ordered, sensor-link-frame positions for a tactile taxel layout.
 * The position order is the taxel identity. */
struct TactileTaxelLayout {
  /* Taxel positions [m] in the sensor link frame. */
  DynamicArray<Real3> positions;
};

/* @brief Dense taxel-force observation for one tactile site. */
struct TactileTaxelForces {
  /* One force [N] per layout taxel, in the sensor link frame. */
  DynamicArray<Real3> forces;
};

/* @brief Flatten taxel forces into a pre-sized row-major Float32 [N, 3] buffer. */
MOCHI_API void FlattenTactileTaxelForcesInto(
    TactileTaxelForces const& forces,
    Span<float> out,
    superdex::Error& error);

/* @brief Force-only aggregation for one tactile site. */
struct TactileAggregateForce {
  /* Sum of the taxel forces [N], in the sensor link frame. */
  Real3 force = {};
};

/* @brief Force-and-torque aggregation for one tactile site. */
struct TactileAggregateWrench {
  /* Sum of the taxel forces [N], in the sensor link frame. */
  Real3 force = {};

  /* Sum of position cross force about @ref referencePoint [N*m], in the sensor link frame. */
  Real3 torque = {};

  /* Point about which @ref torque is resolved [m], in the sensor link frame. */
  Real3 referencePoint = {};
};

/* @brief Configuration for mapping a tactile spherical cap to a rectangle. */
struct TactileHemisphereRectParams {
  /* Hemisphere center [m] in the sensor link frame. */
  Real3 center = {};

  /* Direction from @ref center to the tactile tip. It is normalized by the mapper. */
  Real3 tipDirection{0_r, 0_r, 1_r};

  /* Direction assigned longitude zero. Its component along @ref tipDirection is removed. */
  Real3 zeroLongitudeDirection{1_r, 0_r, 0_r};

  /* Number of rectangular grid columns. */
  int32_t width = 0;

  /* Number of rectangular grid rows. */
  int32_t height = 0;

  /* Largest selected polar angle [rad]. Pi/2 describes a hemisphere. */
  real maxPolarLatitude = 1.5707963267948966_r;

  /* Spherical-cap flattening geometry. */
  TactileRectProjectionType projection = TactileRectProjectionType::AzimuthalEqualArea;

  /* Force-conserving resampling from projected taxels into grid cells. */
  TactileRectRasterizationType rasterization = TactileRectRasterizationType::ConservativeVoronoi;

  /* Linear fine-grid scale used to resolve discrete Voronoi regions before output downsampling. */
  int32_t voronoiOversampling = 4;

  /* Whether source taxels beyond @ref maxPolarLatitude are rejected or explicitly excluded. */
  TactileRectOutsidePolicy outsidePolicy = TactileRectOutsidePolicy::Reject;
};

/* @brief Resolved, immutable mapping from a taxel layout to a rectangular grid. */
struct TactileRectProjectionMap {
  /* Normalized rectangular coordinates in [-1, 1], one per source taxel. */
  DynamicArray<Real2> coordinates;

  /* Row-major destination cell per source taxel, or -1 when explicitly excluded. */
  DynamicArray<int32_t> taxelCells;

  /* CSR offsets into @ref contributionCells and @ref contributionWeights, one row per taxel. */
  DynamicArray<int32_t> taxelContributionOffsets;

  /* Row-major destination cells receiving force from each taxel. */
  DynamicArray<int32_t> contributionCells;

  /* Per-taxel force fractions. Every selected taxel's CSR row sums to one. */
  DynamicArray<real> contributionWeights;

  /* Number of grid columns. */
  int32_t width = 0;

  /* Number of grid rows. */
  int32_t height = 0;

  /* Number of source taxels selected by the projection. */
  int32_t selectedTaxelCount = 0;

  /* Resampling used to build the contribution matrix. */
  TactileRectRasterizationType rasterization = TactileRectRasterizationType::ConservativeVoronoi;

  /* Fine-grid scale used for conservative Voronoi resampling, or one for nearest-cell mode. */
  int32_t voronoiOversampling = 1;
};

/* @brief Rectangular tactile force grid for one site. */
struct TactileForceGrid {
  /* Row-major cell force sums [N] in the sensor link frame. */
  DynamicArray<Real3> forces;

  /* Number of source taxels making a nonzero contribution to each row-major cell. */
  DynamicArray<int32_t> taxelCounts;

  /* Number of grid columns. */
  int32_t width = 0;

  /* Number of grid rows. */
  int32_t height = 0;
};

/* @brief Build an ordered tactile taxel layout.
 *
 * @param positions Taxel positions [m] in sensor link frame, in stable taxel order.
 * @param error Error status. Empty layouts and non-finite positions are rejected.
 * @return Validated layout, or an empty layout on error. */
MOCHI_API TactileTaxelLayout
MakeTactileTaxelLayout(Span<Real3 const> positions, superdex::Error& error);

/* @brief Convert a sparse node-force query into a dense taxel-force observation.
 *
 * @param nodeForces Sparse force records [N] in world frame. Repeated indices are accumulated.
 * @param worldFromSensor Sensor-link pose in world space.
 * @param layout Ordered taxel layout that defines the output shape and identity.
 * @param error Error status. Out-of-range indices and non-finite forces are rejected.
 * @return One sensor-link-frame force per layout taxel, or an empty observation on error. */
MOCHI_API TactileTaxelForces ScatterTactileTaxelForces(
    Span<NodeContactForce const> nodeForces,
    TransformRT const& worldFromSensor,
    TactileTaxelLayout const& layout,
    superdex::Error& error);

/* @brief Scatter into a pre-sized taxel-force observation without allocating.
 * @p out.forces must already have one entry per layout taxel. */
MOCHI_API void ScatterTactileTaxelForcesInto(
    Span<NodeContactForce const> nodeForces,
    TransformRT const& worldFromSensor,
    TactileTaxelLayout const& layout,
    TactileTaxelForces& out,
    superdex::Error& error);

/* @brief Read a dense taxel-force observation from a Mochi actor.
 *
 * The caller must register @ref QueryType::NodeContactForces before stepping the scene and keep
 * that query registered while reading. This function deliberately does not own query lifetime, so
 * applications can share one registration across multiple consumers.
 *
 * @param actor Actor whose node-force query is read.
 * @param worldFromSensor Sensor-frame pose in world space. Pass the actor root transform when the
 * actor and sensor frames coincide.
 * @param layout Ordered taxel layout matching the actor's node indices.
 * @param error Error status.
 * @return One sensor-link-frame force per layout taxel, or an empty observation on error. */
MOCHI_API TactileTaxelForces ReadTactileTaxelForces(
    Actor const* actor,
    TransformRT const& worldFromSensor,
    TactileTaxelLayout const& layout,
    superdex::Error& error);

/* @brief Read into a pre-sized taxel-force observation without allocating.
 * @p out.forces must already have one entry per layout taxel. */
MOCHI_API void ReadTactileTaxelForcesInto(
    Actor const* actor,
    TransformRT const& worldFromSensor,
    TactileTaxelLayout const& layout,
    TactileTaxelForces& out,
    superdex::Error& error);

/* @brief Sum a taxel-force observation into a force-only aggregate.
 *
 * No learned response, calibration, threshold, or torque model is applied.
 *
 * @param taxelForces Dense taxel-force observation.
 * @param error Error status. Empty inputs and non-finite forces are rejected.
 * @return Sum of all taxel forces. */
MOCHI_API TactileAggregateForce
AggregateTactileForce(TactileTaxelForces const& taxelForces, superdex::Error& error);

/* @brief Aggregate taxel forces into a wrench about @p referencePoint.
 *
 * @param layout Ordered taxel positions in the sensor link frame.
 * @param taxelForces Taxel forces indexed by @p layout.
 * @param referencePoint Torque reference point [m] in the sensor link frame.
 * @param error Error status. Mismatched layouts and non-finite inputs are rejected.
 * @return Aggregated force and torque. */
MOCHI_API TactileAggregateWrench AggregateTactileWrench(
    TactileTaxelLayout const& layout,
    TactileTaxelForces const& taxelForces,
    Real3 const& referencePoint,
    superdex::Error& error);

/* @brief Resolve a tactile spherical-cap layout into a rectangular projection.
 *
 * Every projection places the tactile tip at continuous coordinate (0, 0), halfway between the four
 * central cells of an even-sized grid. Nearest-cell rasterization deterministically resolves that
 * tie toward the larger row and column. Taxels outside the configured spherical cap are either
 * rejected or assigned cell -1 according to @ref TactileHemisphereRectParams::outsidePolicy.
 *
 * @param layout Ordered taxel positions in the sensor link frame.
 * @param params Hemisphere frame and output grid shape.
 * @param error Error status. Invalid frames, mismatched layouts and positions outside the requested
 * cap under the reject policy are rejected.
 * @return Stable continuous coordinates, nearest cells and a force-conserving contribution map. */
MOCHI_API TactileRectProjectionMap BuildTactileRectProjection(
    TactileTaxelLayout const& layout,
    TactileHemisphereRectParams const& params,
    superdex::Error& error);

/* @brief Rasterize taxel forces through a resolved tactile projection.
 *
 * Every selected taxel's contribution weights sum to one, so the grid force sum equals the
 * selected input force sum for either rasterization mode.
 *
 * @param projection Resolved mapping built for the source layout.
 * @param taxelForces Taxel forces indexed by that layout.
 * @param error Error status. Mismatched layouts and non-finite forces are rejected.
 * @return Row-major rectangular force grid and per-cell taxel counts. */
MOCHI_API TactileForceGrid RasterizeTactileRectProjection(
    TactileRectProjectionMap const& projection,
    TactileTaxelForces const& taxelForces,
    superdex::Error& error);

/* @brief Rasterize through a resolved projection without allocating.
 * @p out forces and taxel counts must each contain width times height elements. */
MOCHI_API void RasterizeTactileRectProjectionInto(
    TactileRectProjectionMap const& projection,
    TactileTaxelForces const& taxelForces,
    TactileForceGrid& out,
    superdex::Error& error);

/* @brief Resolve and rasterize a taxel-force observation in one call.
 *
 * Prefer caching @ref BuildTactileRectProjection when mapping repeated frames from one layout.
 */
MOCHI_API TactileForceGrid MapTactileHemisphereToRect(
    TactileTaxelLayout const& layout,
    TactileTaxelForces const& taxelForces,
    TactileHemisphereRectParams const& params,
    superdex::Error& error);

/* @brief Compatibility wrapper that resolves and rasterizes into a pre-sized rectangular grid.
 *
 * @deprecated This function rebuilds and allocates the projection on every call. Repeated-frame
 * callers should cache @ref BuildTactileRectProjection and call @ref
 * RasterizeTactileRectProjectionInto, which preserves @p out storage.
 * @p out forces and taxel counts must each contain width times height elements. */
[[deprecated("Cache BuildTactileRectProjection and call RasterizeTactileRectProjectionInto")]]
MOCHI_API void MapTactileHemisphereToRectInto(
    TactileTaxelLayout const& layout,
    TactileTaxelForces const& taxelForces,
    TactileHemisphereRectParams const& params,
    TactileForceGrid& out,
    superdex::Error& error);

} // namespace superdex::robotics

MOCHI_ENUM_BEGIN(superdex::robotics::TactileRectProjectionType)
MOCHI_ENUM_ITEM(AzimuthalEqualArea)
MOCHI_ENUM_ITEM(AzimuthalEquidistant)
MOCHI_ENUM_ITEM(Octahedral)
MOCHI_ENUM_END()

MOCHI_ENUM_BEGIN(superdex::robotics::TactileRectRasterizationType)
MOCHI_ENUM_ITEM(ConservativeVoronoi)
MOCHI_ENUM_ITEM(NearestCell)
MOCHI_ENUM_END()

MOCHI_ENUM_BEGIN(superdex::robotics::TactileRectOutsidePolicy)
MOCHI_ENUM_ITEM(Reject)
MOCHI_ENUM_ITEM(Exclude)
MOCHI_ENUM_END()
