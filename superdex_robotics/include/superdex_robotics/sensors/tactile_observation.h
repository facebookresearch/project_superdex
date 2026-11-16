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

/* @brief Configuration for mapping a hemispherical tactile layout to a rectangle. */
struct TactileHemisphereRectParams {
  /* Hemisphere center [m] in the sensor link frame. */
  Real3 center = {};

  /* Direction from @ref center to the tactile tip. It is normalized by the mapper. */
  Real3 tipDirection{0_r, 0_r, 1_r};

  /* Direction assigned longitude zero. Its component along @ref tipDirection is removed. */
  Real3 zeroLongitudeDirection{1_r, 0_r, 0_r};

  /* Number of rectangular grid columns. */
  int width = 0;

  /* Number of rectangular grid rows. */
  int height = 0;

  /* Largest supported polar latitude [rad]. Pi/2 describes a hemisphere. */
  real maxPolarLatitude = 1.5707963267948966_r;
};

/* @brief Rectangular tactile force grid for one site. */
struct TactileForceGrid {
  /* Row-major cell force sums [N] in the sensor link frame. */
  DynamicArray<Real3> forces;

  /* Number of source taxels accumulated into each row-major cell. */
  DynamicArray<int> taxelCounts;

  /* Number of grid columns. */
  int width = 0;

  /* Number of grid rows. */
  int height = 0;
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

/* @brief Map taxel forces on a hemisphere to a rectangular grid.
 *
 * Polar latitude is zero at the tactile tip and reaches @ref
 * TactileHemisphereRectParams::maxPolarLatitude at the final row. Longitude is zero along @ref
 * TactileHemisphereRectParams::zeroLongitudeDirection and wraps at the left/right grid seam. The
 * tip therefore has continuous coordinates (latitude=0, longitude=0) and rasterizes to the center
 * column of the first row. Each taxel contributes to exactly one cell, so the grid force sum equals
 * the input force sum.
 *
 * @param layout Ordered taxel positions in the sensor link frame.
 * @param taxelForces Taxel forces indexed by @p layout.
 * @param params Hemisphere frame and output grid shape.
 * @param error Error status. Invalid frames, mismatched layouts and positions outside the requested
 * hemisphere are rejected.
 * @return Row-major rectangular force grid and per-cell taxel counts. */
MOCHI_API TactileForceGrid MapTactileHemisphereToRect(
    TactileTaxelLayout const& layout,
    TactileTaxelForces const& taxelForces,
    TactileHemisphereRectParams const& params,
    superdex::Error& error);

/* @brief Map into a pre-sized rectangular grid without allocating.
 * @p out forces and taxel counts must each contain width times height elements. */
MOCHI_API void MapTactileHemisphereToRectInto(
    TactileTaxelLayout const& layout,
    TactileTaxelForces const& taxelForces,
    TactileHemisphereRectParams const& params,
    TactileForceGrid& out,
    superdex::Error& error);

} // namespace superdex::robotics
