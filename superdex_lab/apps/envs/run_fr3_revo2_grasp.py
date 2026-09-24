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
pre-shape the thumb, descend over the object, close the hand, and lift. The arm follows
inverse kinematics solved on a collider-free twin of the robot. Each phase prints the
five fingertip pads' normal forces, and the final taxel maps are printed at the end.

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
CLOSED_HAND = np.array([0.8, 0.9, 0.45, 0.45, 0.45, 0.45])

# (phase, duration [s], grasp-point height above the object center [m], hand target).
# Grasping 2 cm above the center keeps the opposed thumb clear of the table.
PHASES = (
    ("pre-shape", 0.6, 0.18, OPEN_HAND),
    ("descend", 1.6, 0.02, OPEN_HAND),
    ("close", 1.2, 0.02, CLOSED_HAND),
    ("lift", 1.6, 0.17, CLOSED_HAND),
    ("hold", 1.0, 0.17, CLOSED_HAND),
)


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
    for name, duration, height, hand in PHASES:
        steps = int(round(duration * env.get_control_frequency()))
        start = env.grasp_point()
        goal = target + np.array([0.0, 0.0, height])
        for k in range(steps):
            # Straight-line path of the grasp point, tracked through IK.
            waypoint = start + (goal - start) * min(1.0, (k + 1) / (0.8 * steps))
            arm_q = ik.solve(env.arm_target(), waypoint)
            arm = np.clip((arm_q - env.arm_target()) / step_limit, -1.0, 1.0)
            action = env.to_action(
                {"arm": arm.astype(np.float32), "hand": hand.astype(np.float32)}
            )
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print(f"  stopped: {info.get('terminated_reason', 'truncated')}")
                return False
        forces = " ".join(
            f"{f}={info['tactile_normal_force'][f]:5.2f}N" for f in FINGERS
        )
        print(
            f"  {name:9s} | lifted {100 * info['object_height']:5.1f} cm | "
            f"pads: {forces} | reward {reward:6.2f}"
        )
    obs = env.to_structured_observation(obs)
    offset = 0
    for finger in FINGERS:
        rows, cols = env._sensors[finger].shape
        taxels = obs["taxels"][offset : offset + rows * cols].reshape(rows, cols)
        offset += rows * cols
        if taxels.any():
            print(f"  {finger} taxels [N] (rows along the finger):")
            print(
                "    "
                + np.array2string(taxels, precision=2, suppress_small=True).replace(
                    "\n", "\n    "
                )
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
