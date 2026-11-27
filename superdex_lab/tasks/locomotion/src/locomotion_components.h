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

namespace superdex::lab {

class LabKinds;

/// Registers the components of velocity-tracking tasks (see ComponentKind):
/// - `twist_command` {"base" (the robot's base link, e.g. "pelvis"), "linVelX", "linVelY",
///   "angVelZ" ([lo, hi] each, m/s and rad/s), "heading" ([lo, hi] rad), "resamplingS" ([lo, hi]
///   s), "standingShare", "headingShare", "headingCommand", "headingStiffness", "minNorm",
///   "phasePeriodS" (s), all required; "linVelXGlobals", "linVelYGlobals", "angVelZGlobals"
///   (optional [lo, hi] global names)}: unitree_rl_mjlab's UniformVelocityCommand. Each resample,
///   after a period drawn from resamplingS (s), draws the base-frame command (vx, vy, wz) uniformly
///   from the ranges (from the named globals' values instead, when given: a curriculum widens
///   them), zeroes it when its norm is at most minNorm, and draws a heading target and whether the
///   env is a heading env (headingShare) and a standing env (standingShare). Each Update then sets
///   a heading env's wz to clip(headingStiffness * the heading error, angVelZ) (headingCommand) and
///   a standing env's command to zero. The heading is the base link's x axis's yaw in the world.
///   The heading clip uses the wz range of the last resample (the globals' values then); upstream
///   clips with its cfg's range, which its curriculum edits at once. Upstream's
///   init_velocity_prob (default 0) is not ported.
///   Reset draws the first command; the env's update after a reset (advancing no time) then applies
///   the heading and standing rules, as mjlab's command compute after its resets. Outputs:
///   "command" (vx, vy, wz) and "phase" (sin, cos of 2 pi t / phasePeriodS, t the episode's time;
///   zeros when |command| < 0.1), mjlab velocity's `phase` observation. With "schedule"
///   ([[t, vx, vy, wz], ...], t in s from the episode's start, the first at 0, increasing), the
///   command is the last row whose time has come, with no draws, heading control or standing envs:
///   fixed commands for clips and evaluations.
/// - `foot_contact` {"links" (one link set per foot, e.g. "links:left_foot?_s?"), "sites" (one
///   site link per foot, e.g. "left_foot"), "ground" (the scene actor, e.g. "StaticPlane"),
///   "base" (the robot's base link), "forceThreshold" (N), all required}: mjlab's ContactSensor
///   with track_air_time on the feet against the ground, updated once per decision in Sense (right
///   after the physics, so the terms of the same step read it). Per foot: "force", the summed
///   norms of the contact forces between the foot's links and the ground (N; mjlab's net force's
///   norm), and "force_log1p", log(1 + force) (mjlab's foot_contact_forces observation); "found", 1
///   when the force exceeds forceThreshold; "contact_time" and "air_time", how long (s) the foot
///   has been in contact or in the air (0 otherwise); "last_air_time", the air time its last
///   landing ended; "first_contact", 1 on the decision it landed (mjlab's compute_first_contact);
///   "height", the site's height (m, world z: mjlab's foot_height); and "speed_xy", the site's
///   speed along the world's x and y axes (m/s: mjlab's site_lin_vel_w). Reset zeroes the times and
///   reads the heights and speeds; contacts start being read at the first step.
/// - `push_robot` (an event) {"intervalS" ([lo, hi] s), "linear", "angular" ([lo, hi] per world
///   axis, m/s and rad/s), all required}: mjlab's push_by_setting_velocity: every interval drawn
///   from intervalS (drawn again at each reset), adds a twist drawn uniformly per axis to the
///   robot's root velocity (StepContext::PushRobot). Output "velocity", the last push's (linear,
///   angular), held until the next push.
void RegisterLocomotionComponents(LabKinds& kinds);

} // namespace superdex::lab
