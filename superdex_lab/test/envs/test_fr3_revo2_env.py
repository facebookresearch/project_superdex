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

"""Behavior tests for the FR3 + Revo2 tactile environment (``Fr3Revo2Env``)."""

from __future__ import annotations

import importlib.util
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import superdex.physics as physics
from superdex.lab.gym.envs.robots.fr3_revo2_env import (
    ARM_HOME,
    FINGERS,
    HAND_JOINT_SPEEDS,
    Fr3Revo2Env,
    Fr3Revo2EnvCfg,
)
from superdex.lab.gym.utils import mochi_helpers
from superdex.physics.paths import get_assets_root

_GRASP_DEMO = (
    Path(__file__).resolve().parents[2] / "apps" / "envs" / "run_fr3_revo2_grasp.py"
)


def _load_grasp_demo():
    spec = importlib.util.spec_from_file_location("run_fr3_revo2_grasp", _GRASP_DEMO)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _action(env, arm=None, hand=None):
    return env.to_action(
        {
            "arm": np.zeros(7, np.float32)
            if arm is None
            else np.asarray(arm, np.float32),
            "hand": -np.ones(6, np.float32)
            if hand is None
            else np.asarray(hand, np.float32),
        }
    )


class TestFr3Revo2Env(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.env = Fr3Revo2Env(Fr3Revo2EnvCfg(steps_per_episode=-1))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    def test_spaces(self) -> None:
        structure = self.env.get_observation_space_structure()
        # Per finger, the Revo2 Touch channels: normal, tangential, cos, sin, proximity.
        self.assertEqual(structure["tactile"].shape, (len(FINGERS) * 5,))
        self.assertNotIn("taxels", structure)
        self.assertEqual(self.env.action_space.shape, (13,))

    def test_reset_is_seeded_and_starts_at_home(self) -> None:
        first, _ = self.env.reset(seed=3)
        second, _ = self.env.reset(seed=3)
        np.testing.assert_array_equal(first, second)
        obs = self.env.to_structured_observation(first)
        np.testing.assert_allclose(obs["arm_qpos"], ARM_HOME["right"], atol=1e-4)
        np.testing.assert_allclose(obs["hand_qpos"], np.zeros(6), atol=1e-4)
        cfg = self.env._cfg
        offset = np.abs(obs["object_pos"][:2] - np.asarray(cfg.object_xy))
        self.assertTrue(np.all(offset <= cfg.object_xy_noise + 1e-6))

    def test_object_rests_on_the_table(self) -> None:
        obs, info = self.env.reset(seed=0)
        for _ in range(5):
            obs, _, _, _, info = self.env.step(_action(self.env))
        # The paper cup is 11.2 cm tall and stands on the tabletop at z = 0.
        self.assertAlmostEqual(
            float(self.env.to_structured_observation(obs)["object_pos"][2]),
            0.056,
            delta=0.003,
        )
        self.assertLess(abs(info["object_height"]), 0.002)

    def test_idle_hold_senses_nothing(self) -> None:
        self.env.reset(seed=0)
        for _ in range(10):
            obs, _, terminated, _, info = self.env.step(_action(self.env))
        obs = self.env.to_structured_observation(obs)
        self.assertFalse(terminated)
        np.testing.assert_allclose(obs["arm_qpos"], ARM_HOME["right"], atol=2e-3)
        self.assertEqual(info["fingers_in_contact"], [])
        tactile = obs["tactile"].reshape(len(FINGERS), 5)
        # No force and nothing within the 1 cm proximity range.
        np.testing.assert_array_equal(tactile[:, [0, 1, 4]], 0.0)

    def test_hand_respects_rated_joint_speeds(self) -> None:
        self.env.reset(seed=0)
        obs, *_ = self.env.step(_action(self.env, hand=np.ones(6)))
        moved = self.env.to_structured_observation(obs)["hand_qpos"]
        limit = np.asarray(HAND_JOINT_SPEEDS) * self.env.get_control_timestep()
        self.assertTrue(np.all(np.abs(moved) <= limit + 1e-3), (moved, limit))

    def test_mounted_hand_keeps_its_mimic_couplings(self) -> None:
        # Five couplings: superdex-robotics 1.0.0 drops them when a hand is attached to
        # an arm, so this guards the workaround in bot_loading.
        self.assertEqual(len(self.env._mimic), 5)
        self.env.reset(seed=0)
        for _ in range(15):
            self.env.step(_action(self.env, hand=[-1, -1, 0, 0, 0, 0]))
        q = mochi_helpers.get_articulated_pose(self.env._agent)
        for follower, leader, multiplier in self.env._mimic:
            if q[leader] > 0.2:
                self.assertAlmostEqual(q[follower] / q[leader], multiplier, delta=0.03)

    def test_scripted_grasp_lifts_with_tactile_contact(self) -> None:
        demo = _load_grasp_demo()
        ik = demo.ArmIK("right")
        try:
            with redirect_stdout(io.StringIO()):
                success = demo.run_episode(self.env, ik, seed=0)
        finally:
            ik.close()
        info = self.env._last_info
        self.assertTrue(success)
        self.assertGreater(info["object_height"], 0.1)
        self.assertIn("thumb", info["fingers_in_contact"])
        self.assertGreaterEqual(len(info["fingers_in_contact"]), 2)
        self.assertGreater(self.env._last_reward["lift"], 0.0)
        # The tactile grip regulates each touching fingertip near its target force,
        # within the sensor's 0-25 N range, and a touching pad reads full proximity.
        for finger in info["fingers_in_contact"]:
            self.assertLess(info["tactile_normal_force"][finger], 10.0)
            self.assertEqual(info["tactile_proximity"][finger], 1.0)


class TestFr3Revo2EnvVariants(unittest.TestCase):
    def test_tactile_noise_is_seeded_per_env(self) -> None:
        cfg = Fr3Revo2EnvCfg(tactile_noise_std=0.2)

        def forces(env, seed):
            obs = env.to_structured_observation(env.reset(seed=seed)[0])
            return obs["tactile"].reshape(len(FINGERS), 5)[:, :2]

        with Fr3Revo2Env(cfg) as env_1, Fr3Revo2Env(cfg) as env_2:
            # The two envs share one scene and its sensors; noise must still be per env.
            self.assertIs(env_1._scene, env_2._scene)
            first = forces(env_1, 1)
            forces(env_2, 2)
            again = forces(env_1, 1)
            self.assertTrue(first.any())  # noise shows up even without contact
            self.assertTrue(np.all(first >= 0))
            # Still quantized to the sensor's 0.1 N resolution.
            np.testing.assert_allclose(first * 10, np.round(first * 10), atol=1e-4)
            np.testing.assert_array_equal(first, again)

    def test_batched_stepping_on_a_shared_scene_matches_a_lone_env(self) -> None:
        """Interleaved envs on one scene reproduce a lone env's trajectory exactly, so
        per-env state (controller targets, observations) does not leak between them."""
        rng = np.random.default_rng(7)
        actions = [rng.uniform(-1, 1, 13).astype(np.float32) for _ in range(12)]

        def final_obs(env):
            return env.get_last_step().observation

        with Fr3Revo2Env(Fr3Revo2EnvCfg(use_shared_scenes=False)) as reference:
            reference.reset(seed=5)
            for action in actions:
                reference.step(action)
            expected = final_obs(reference)
        with (
            Fr3Revo2Env(Fr3Revo2EnvCfg()) as env_1,
            Fr3Revo2Env(Fr3Revo2EnvCfg()) as env_2,
        ):
            env_1.reset(seed=5)
            env_2.reset(seed=5)
            for i in range(0, len(actions), 4):
                for action in actions[i : i + 4]:
                    env_1.step(action)
                for action in actions[i : i + 4]:
                    env_2.step(action)
            for env in (env_1, env_2):
                for key in ("arm_qpos", "hand_qpos", "object_pos", "tactile"):
                    np.testing.assert_array_equal(
                        final_obs(env)[key], expected[key], key
                    )

    def test_box_object(self) -> None:
        with Fr3Revo2Env(Fr3Revo2EnvCfg(object_name="box")) as env:
            obs, _ = env.reset(seed=0)
            center_z = env.to_structured_observation(obs)["object_pos"][2]
            self.assertAlmostEqual(float(center_z), 0.045, delta=0.003)


class TestRevo2Colliders(unittest.TestCase):
    def test_collider_sdfs_have_consistent_signs(self) -> None:
        """A signed distance changes by at most one voxel between neighboring grid
        samples. A sample with the wrong sign breaks that (the value jumps from -d to
        +d), and makes contact pull objects into the link, or let them sink into it."""
        assets = Path(get_assets_root())
        paths = sorted(
            [
                *assets.glob("bots/hands/revo2/*/collision/*.mochi.h5"),
                *assets.glob("bots/arm_hand_combos/fr3_v2_revo2/collision/*.mochi.h5"),
            ]
        )
        self.assertEqual(len(paths), 35)  # 17 links per hand, and the FR3 mount
        for path in paths:
            with self.subTest(collider=path.name):
                sdf = physics.model.load_from_file(str(path)).sdf
                dims = tuple(int(d) for d in sdf.dims)
                values = np.asarray(sdf.values, dtype=np.float64).reshape(dims)
                lo = np.asarray(sdf.bounds.min, dtype=np.float64)
                hi = np.asarray(sdf.bounds.max, dtype=np.float64)
                voxel = (hi - lo) / (np.asarray(dims) - 1)
                for axis in range(3):
                    step = np.abs(np.diff(values, axis=axis)).max() / voxel[axis]
                    self.assertLessEqual(step, 1.01, f"axis {axis}")


if __name__ == "__main__":
    unittest.main()
