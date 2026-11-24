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

#include "superdex_asset_tuner/tuner_spec.h"

#include <array>

namespace superdex::tuner {

// Scalar load magnitude at time `t`. Keyframes profiles need their keys sorted by time (see
// SortByTime).
[[nodiscard]] real EvaluateProfile(LoadProfile const& profile, double t);

// Whether the load's direction (Force, Torque, JointEffort weights) or application point (Force)
// comes from its keyframes: a Keyframes profile with keys and the matching animate switch on.
[[nodiscard]] bool AnimatesDirection(LoadSpec const& load);
[[nodiscard]] bool AnimatesPoint(LoadSpec const& load);

// The load's direction at `t`, in its own frame: `load.direction`, or when animated the keys'
// directions turned along the shortest arc between keys (unit length), or for JointEffort the
// keys' weights blended as values. Keys must be sorted by time.
[[nodiscard]] Real3 EvaluateLoadDirection(LoadSpec const& load, double t);

// The load's application point at `t`, in its frame's coordinates (body-local, or world for
// WorldPoint): the load's own point, or when animated the keys' points. Keys must be sorted.
[[nodiscard]] Real3 EvaluateLoadPoint(LoadSpec const& load, double t);

// Smallest rotation taking `from` onto `to` (either may be unnormalized). Identity if either is
// zero; half a turn about an axis perpendicular to `from` when they are opposite.
[[nodiscard]] mochi::Quaternion ShortestRotation(Real3 const& from, Real3 const& to);

// Target pose on a path at time `t`: the load's point (translation) and the body's orientation
// (rotation). Before the first waypoint it is the first's pose, after the last the last's.
// Waypoints must be sorted by time and non-empty.
[[nodiscard]] mochi::TransformRT EvaluatePosePath(Span<PoseWaypoint const> waypoints, double t);

// The keys' values at time `t`, one per entry of `out`, interpolated as profile keys are. Keys must
// be sorted by time, non-empty, and hold at least out.size() values each.
void EvaluateInputKeys(Span<InputKey const> keys, double t, Span<real> out);

// Drive mode at time `t`: that of the last waypoint at or before `t`, or the first's before it.
// Waypoints must be sorted by time and non-empty.
[[nodiscard]] PathDriveMode EvaluatePathDrive(Span<PoseWaypoint const> waypoints, double t);

// Average of ixx, iyy and izz from an inertia stored as [ixx, ixy, ixz, iyy, iyz, izz]. One number
// that stands for the body's inertia.
[[nodiscard]] real MeanDiagonalInertia(mochi::Real6 const& inertia);

// The mass [kg] and inertia [kg m^2] that set how strong a PosePath drive is.
struct DriveScale {
  real mass = 1;
  real inertia = 0;
};

// The drive scale for a body with `ownMass` and `ownInertia`. For a link, `articulationMass` is the
// mass of the whole articulation (0 for a rigid body). A link uses that mass, which lets a light
// link at the end of a chain still move the rest of it. Its inertia grows by the same factor. A
// body with no mass counts as 1 kg.
[[nodiscard]] DriveScale MakeDriveScale(real ownMass, real ownInertia, real articulationMass);

// Stiffness, damping and saturation for a spring-damper on `mass` that responds at `hz` with
// `dampingRatio`. The spring's pull is limited to mass * maxAcceleration. Saturation is -1 (no
// limit) when maxAcceleration is 0. All zero when `hz` is 0 or less.
[[nodiscard]] std::array<real, 3>
SpringGains(real mass, real hz, real dampingRatio, real maxAcceleration);

// How hard a PosePath drive pulls on a body that cannot move at all. This is the most it can pull.
// The spring reaches springForce once the body is capLag behind the path, and stays there. The
// damper adds more on top, based on how fast the path moves. It drops to 0 when the path stops.
struct DrivePullEstimate {
  // [N] and [m]. Both -1 when there is no limit and the spring keeps growing as the body falls
  // behind.
  real springForce = -1;
  real capLag = -1;
  // The path's highest velocity while the drive is on [m/s],
  // and the damper's pull at that velocity [N].
  real peakVelocity = 0;
  real damperForce = 0;
  // The same for turning: [N m], [rad], [rad/s] and [N m]. Torques are 0 when turning is not
  // driven.
  real springTorque = -1;
  real capAngle = -1;
  real peakAngularVelocity = 0;
  real damperTorque = 0;
};

// Estimates how hard `load`'s drive pulls on a body with `scale`. `load` is a PosePath with its
// waypoints sorted by time. `dt` is the step size [s]. A Step waypoint's jump is treated as
// happening within one step.
[[nodiscard]] DrivePullEstimate
EstimateDrivePull(LoadSpec const& load, DriveScale const& scale, double dt);

// Sorts the load's waypoints, profile keys and input keys by time, keeping the order of equal
// times.
void SortByTime(LoadSpec& load);

// Rigid actor external-force DoFs (world force, world torque about the center of mass) equivalent
// to `force` applied at `pointWorld`.
[[nodiscard]] std::array<real, 6>
RigidPointForceToDofs(Real3 const& force, Real3 const& pointWorld, Real3 const& comWorld);

// Adds J^T [force; (pointWorld - jacobianRefWorld) x force] to `inOutGeneralized`. `jacobian` is
// row-major 6 x numDofs (3 translation rows, then 3 rotation rows) in world frame, with its
// translation rows describing the velocity of `jacobianRefWorld`.
void AddLinkForceToGeneralized(
    Span<real const> jacobian,
    Real3 const& force,
    Real3 const& pointWorld,
    Real3 const& jacobianRefWorld,
    Span<real> inOutGeneralized);

// Adds J^T [0; torque] to `inOutGeneralized`, with `jacobian` as in AddLinkForceToGeneralized.
void AddLinkTorqueToGeneralized(
    Span<real const> jacobian,
    Real3 const& torque,
    Span<real> inOutGeneralized);

} // namespace superdex::tuner
