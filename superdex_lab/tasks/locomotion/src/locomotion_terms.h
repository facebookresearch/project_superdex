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

/// Registers the velocity-tracking terms, named as unitree_rl_mjlab's (src/tasks/velocity/mdp):
/// - `track_linear_velocity` {"command" (a twist_command), "velocity" (a 3-wide feature: the base's
///   linear velocity in base axes, e.g. "pelvis.linear_velocity_from_base"), "std"}:
///   exp(-((cx - vx)² + (cy - vy)² + 2 vz²) / std²).
/// - `track_angular_velocity` {"command", "velocity" (the base's angular velocity in base axes),
///   "std"}: exp(-((cz - wz)² + 0.05 (wx² + wy²)) / std²).
///
/// The foot terms read a foot_contact ("contact"), and are active only while the command's
/// |(vx, vy)| + |wz| exceeds "commandThreshold" (default 0.1); `stand_still` only while it does
/// not:
/// - `feet_gait` {"command", "contact", "period" (s), "offset" (one per foot, cycles),
/// "threshold"}:
///   the share of feet whose contact matches the gait clock's stance (phase (t / period + offset)
///   mod 1 below threshold, t the episode's decisions times the control step).
/// - `feet_clearance` {"command", "contact", "targetHeight" (m)}: the sum over feet of |height -
///   targetHeight| times speed_xy.
/// - `feet_slip` {"command", "contact"}: the sum over feet in contact of speed_xy².
/// - `soft_landing` {"command", "contact"}: the sum over feet that landed of their force.
/// - `variable_posture` {"command", "joints" (a joint offset feature, e.g.
/// "body.joints.from_initial"),
///   "stdStanding", "stdWalking", "stdRunning" (one per joint), "walkingThreshold",
///   "runningThreshold"}: exp(-mean(offset² / std²)), the std by the command's speed: standing
///   below walkingThreshold, running from runningThreshold on.
/// - `stand_still` {"command", "joints"}: the sum of the joint offsets squared.
///
/// With "body.joints.from_initial" as "joints", the offsets are from the joints' reset positions,
/// where upstream's are from its default_joint_pos: the same while a lab's reset adds no joint
/// offsets (the G1 labs').
/// - `is_terminated` {}: 1 on a step that ended the episode as a failure (not a time limit).
/// - `projected_gravity_xy` {"rotation" (a link's rotation quaternion in the world, e.g.
///   "torso_link.rotation_in_task_root.quat")}: the squared x and y of the unit gravity in the
///   link's axes: mjlab's body_orientation_l2.
/// - `body_angular_velocity_xy` {"velocity" (a link's angular velocity in base axes),
/// "baseRotation"
///   (the base's rotation quaternion in the world)}: the squared world x and y of the angular
///   velocity: mjlab's body_angular_velocity_penalty.
/// - `joint_acc_l2` {"velocities" (a joint velocity feature)}: the sum of the joints' accelerations
///   squared, from their velocities one decision apart (0 at an episode's first decision).
void RegisterLocomotionTerms(LabKinds& kinds);

} // namespace superdex::lab
