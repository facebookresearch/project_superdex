# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Scripted grasp-and-lift on the FR3 + BrainCo Revo2 environment, with tactile readout.

A hand-written policy drives ``Fr3Revo2Env`` through its regular action interface:
pre-shape the thumb, descend over the object, close the hand under tactile force control
(each fingertip closes until its sensor reads ~3 N, then holds that force), and lift. The arm follows
inverse kinematics solved on a collider-free twin of the robot. Each phase prints the
five fingertips' normal forces, and the full Revo2 Touch readings (normal and
tangential force, direction, proximity) are printed at the end.

Usage:
    python run_fr3_revo2_grasp.py [--episodes 3] [--side right] [--render]
"""

from __future__ import annotations

import argparse

import numpy as np
import superdex.physics as physics
import superdex.robotics as robotics
from scipy.spatial.transform import Rotation
from superdex.lab.gym.envs.robots.fr3_revo2_env import (
    COMBO_BOT,
    FINGERS,
    GRASP_POINT_IN_HAND,
    Fr3Revo2Env,
    Fr3Revo2EnvCfg,
)
from superdex.lab.gym.utils import mochi_helpers

# Hand targets in action units ([-1, 1] over each joint's range), in action order:
# thumb metacarpal, thumb flexion, index, middle, ring, pinky.
OPEN_HAND = np.array([0.8, -1.0, -1.0, -1.0, -1.0, -1.0])  # thumb pre-opposed
CLOSED_HAND = np.array([0.8, 0.9, 0.45, 0.45, 0.45, 0.45])  # closing limit

# Tactile grip control, like the Revo2 Touch's force-adaptive grip: each flexing
# actuator closes fast while its fingertip senses nothing, slows down once proximity
# reports the object near, and then regulates the fingertip's normal force.
GRIP_FORCE = 3.0  # [N] per fingertip
GRIP_GAIN = 0.01  # action units per step, per newton of force error
FAST_CLOSE = 0.08  # action units per step, nothing sensed
SLOW_CLOSE = 0.02  # action units per step, object within proximity range
MAX_OPEN = 0.02  # action units per step, when squeezing too hard
# Fingertip read by each actuator, in action order (the metacarpal only rotates the
# thumb into opposition and stays put).
GRIP_FINGERS = (None, "thumb", "index", "middle", "ring", "pinky")

# (phase, duration [s], grasp-point height above the object center [m], grip?).
# Grasping 3.5 cm above the center keeps the opposed thumb clear of the table and the
# fingertips below the cup's widening rim. The close
# phase ends early once the grip holds (see grip_established).
PHASES = (
    ("pre-shape", 0.6, 0.18, False),
    ("descend", 1.6, 0.035, False),
    ("close", 2.0, 0.035, True),
    ("lift", 1.6, 0.185, True),
    ("hold", 1.0, 0.185, True),
)


def regulate_grip(
    hand: np.ndarray, normal_force: dict[str, float], proximity: dict[str, float]
) -> np.ndarray:
    """One step of per-finger proximity-and-force control on the hand command."""
    hand = hand.copy()
    for j, finger in enumerate(GRIP_FINGERS):
        if finger is None:
            continue
        if normal_force[finger] > 0.0:
            step = GRIP_GAIN * (GRIP_FORCE - normal_force[finger])
            hand[j] += np.clip(step, -MAX_OPEN, SLOW_CLOSE)
        else:
            hand[j] += SLOW_CLOSE if proximity[finger] > 0.5 else FAST_CLOSE
    return np.clip(hand, OPEN_HAND, CLOSED_HAND)


def grip_established(normal_force: dict[str, float]) -> bool:
    """The thumb and at least two fingers press with most of the target force."""
    pressing = [f for f in GRIP_FINGERS[1:] if normal_force[f] >= 0.75 * GRIP_FORCE]
    return "thumb" in pressing and len(pressing) >= 3


class ArmIK:
    """Inverse kinematics on a collider-free, gravity-free twin of the robot."""

    def __init__(self, side: str):
        prefab = robotics.load_bot_prefab_from_file(
            str(mochi_helpers.resolve_bot_asset(COMBO_BOT.format(side=side)))
        )
        for i in range(len(prefab.links)):
            prefab.links[i].has_gravity = False
            prefab.links[i].collider_type = physics.ColliderType.NONE
        self.scene = physics.create_scene(f"fr3_revo2_ik_{side}")
        self.bot = robotics.create_bot(self.scene, prefab, robotics.create_context())
        self.actor = self.bot.get_articulated_actor()
        self.wrist = next(
            h
            for h in self.actor.get_nested_link_actors()
            if self.scene.get_actor(h).get_name().endswith(f"/{side}_base_link")
        )
        self.solver = physics.experimental.create_ik_solver(self.scene)
        self.position = self.solver.create_position_target(
            self.wrist, list(GRASP_POINT_IN_HAND), [0.0, 0.0, 0.0], 1e4
        )
        self._pose = physics.DynamicArrayReal(self.actor.get_num_dofs())

    def hold_orientation(self, arm_q: np.ndarray) -> None:
        """Keep the wrist orientation it has at ``arm_q`` in later solves."""
        self._set_arm(arm_q)
        T = self.scene.get_actor(self.wrist).get_root_transform()
        rotvec = Rotation.from_quat(list(T.rotation)).as_rotvec()
        self.solver.create_rotation_target(
            self.wrist, [0.0, 0.0, 0.0], rotvec.tolist(), 1e2
        )

    def solve(self, arm_q: np.ndarray, grasp_point: np.ndarray) -> np.ndarray:
        """Arm joints placing the hand's grasp point at ``grasp_point``, seeded at arm_q."""
        self._set_arm(arm_q)
        self.position.set_target_position(grasp_point.tolist())
        for _ in range(10):
            self.solver.solve_ik()
        self.actor.get_articulated_pose(self._pose)
        return np.array(self._pose)[:7]

    def _set_arm(self, arm_q: np.ndarray) -> None:
        pose = np.zeros(self.actor.get_num_dofs(), dtype=np.float32)
        pose[:7] = arm_q
        self.actor.set_articulated_pose_from_joints(pose)

    def close(self) -> None:
        self.solver.clear_position_target(self.wrist)
        self.solver.clear_rotation_target(self.wrist)
        robotics.destroy_bot(self.scene, self.bot)
        physics.experimental.destroy_ik_solver(self.solver)


def run_episode(env: Fr3Revo2Env, ik: ArmIK, seed: int) -> bool:
    obs, info = env.reset(seed=seed)
    target = env.to_structured_observation(obs)["object_pos"]
    ik.hold_orientation(env.arm_target())
    step_limit = env._cfg.arm_joint_step
    print(f"\nEpisode {seed}: object at ({target[0]:.3f}, {target[1]:.3f}) m")
    hand = OPEN_HAND.copy()
    for name, duration, height, grip in PHASES:
        steps = int(round(duration * env.get_control_frequency()))
        start = env.grasp_point()
        goal = target + np.array([0.0, 0.0, height])
        settled = 0
        for k in range(steps):
            # Straight-line path of the grasp point, tracked through IK.
            waypoint = start + (goal - start) * min(1.0, (k + 1) / (0.8 * steps))
            arm_q = ik.solve(env.arm_target(), waypoint)
            arm = np.clip((arm_q - env.arm_target()) / step_limit, -1.0, 1.0)
            if grip:
                hand = regulate_grip(
                    hand, info["tactile_normal_force"], info["tactile_proximity"]
                )
            action = env.to_action(
                {"arm": arm.astype(np.float32), "hand": hand.astype(np.float32)}
            )
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print(f"  stopped: {info.get('terminated_reason', 'truncated')}")
                return False
            # Lift only once the grip has held for 0.2 s.
            if name == "close":
                settled = (
                    settled + 1 if grip_established(info["tactile_normal_force"]) else 0
                )
                if settled >= 5:
                    break
        forces = " ".join(
            f"{f}={info['tactile_normal_force'][f]:4.1f}N" for f in FINGERS
        )
        print(
            f"  {name:9s} | lifted {100 * info['object_height']:5.1f} cm | "
            f"normal force: {forces} | reward {reward:6.2f}"
        )
    # Final Revo2 Touch readings: per finger [normal, tangential, cos, sin, proximity].
    tactile = env.to_structured_observation(obs)["tactile"].reshape(len(FINGERS), 5)
    print("  fingertip  normal[N]  tangential[N]  direction[deg]  proximity")
    for finger, (normal, tangential, c, s_, proximity) in zip(
        FINGERS, tactile, strict=True
    ):
        direction = np.degrees(np.arctan2(s_, c))
        print(
            f"  {finger:9s} {normal:9.1f} {tangential:14.1f} {direction:15.0f} "
            f"{proximity:10.2f}"
        )
    print(f"  success: {info['is_success']}")
    return bool(info["is_success"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--side", choices=("right", "left"), default="right")
    parser.add_argument("--render", action="store_true", help="open the viewer")
    args = parser.parse_args()

    env = Fr3Revo2Env(
        Fr3Revo2EnvCfg(
            hand_side=args.side,
            render_mode="human" if args.render else None,
            steps_per_episode=-1,
        )
    )
    ik = ArmIK(args.side)
    successes = sum(run_episode(env, ik, seed) for seed in range(args.episodes))
    print(f"\n{successes}/{args.episodes} successful lifts")
    ik.close()
    env.close()


if __name__ == "__main__":
    main()
